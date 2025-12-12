import os
import uuid
import sys
import shutil
import subprocess
import time
import inspect
import textwrap
import base64
import re
from functools import wraps
import dill as pickle
import anndata
from IPython.display import display, Image as IPImage

# ==========================================
#   配置区域
# ==========================================
TEMP_DIR = os.path.join(os.getcwd(), "tmp_interchange")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)

# 定义协议标记
IMG_START_TAG = "###__IMG_BASE64_START__###"
IMG_END_TAG = "###__IMG_BASE64_END__###"
RES_START_TAG = "###__RESULT_BASE64_START__###"
RES_END_TAG = "###__RESULT_BASE64_END__###"

# ==========================================
#   1. SmartAnnData 定义 (保留以备手动使用)
# ==========================================
class SmartAnnData:
    """
    智能 AnnData 包装器 (磁盘版)
    如果手动使用此类包装 AnnData，则强制走磁盘传输。
    """
    def __init__(self, adata_or_path, mode='r+'):
        if isinstance(adata_or_path, str):
             self.adata = None
             self.temp_path = adata_or_path
             self.is_loaded = False
        else:
             self.adata = adata_or_path
             self.temp_path = None
             self.is_loaded = True
        self.mode = mode

    def save_to_disk(self):
        if not self.temp_path or not os.path.exists(self.temp_path):
            unique_id = str(uuid.uuid4())
            self.temp_path = os.path.abspath(os.path.join(TEMP_DIR, f"adata_{unique_id}.h5ad"))
            try:
                self.adata.write_h5ad(self.temp_path)
            except Exception as e:
                print(f"❌ Failed to write .h5ad to {self.temp_path}")
                raise e
        return self.temp_path

    def load_from_disk(self):
        if not self.is_loaded:
            if not self.temp_path or not os.path.exists(self.temp_path):
                raise FileNotFoundError(f"Result file missing on disk: {self.temp_path}")
            import anndata
            self.adata = anndata.read_h5ad(self.temp_path)
            self.is_loaded = True
        return self.adata

    def cleanup(self):
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
            except OSError:
                pass

# ==========================================
#   2. 动态生成远程脚本 (AnnData 内存传输版)
# ==========================================
def _get_remote_script_template():
    
    smart_class_source = f"""
TEMP_DIR = "{TEMP_DIR}"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)

class SmartAnnData:
    def __init__(self, adata_or_path, mode='r+'):
        if isinstance(adata_or_path, str):
             self.adata = None
             self.temp_path = adata_or_path
             self.is_loaded = False
        else:
             self.adata = adata_or_path
             self.temp_path = None
             self.is_loaded = True
        self.mode = mode

    def save_to_disk(self):
        if not self.temp_path:
            import uuid, os
            unique_id = str(uuid.uuid4())
            self.temp_path = os.path.join(TEMP_DIR, f"adata_res_{{unique_id}}.h5ad")
            print(f"💾 [Remote] Saving Result: {{self.temp_path}} ...", flush=True)
            self.adata.write_h5ad(self.temp_path)
        return self.temp_path

    def load_from_disk(self):
        if not self.is_loaded:
            import anndata, os
            if not self.temp_path or not os.path.exists(self.temp_path):
                 raise FileNotFoundError(f"Input file missing: {{self.temp_path}}")
            self.adata = anndata.read_h5ad(self.temp_path)
            self.is_loaded = True
        return self.adata
"""
    
    graphics_patch_source = f"""
import io
import base64
IMG_START_TAG = "{IMG_START_TAG}"
IMG_END_TAG = "{IMG_END_TAG}"

def patch_matplotlib():
    try:
        import matplotlib
        matplotlib.use('Agg') 
        import matplotlib.pyplot as plt
        
        _original_show = plt.show

        def custom_show(*args, **kwargs):
            buf = io.BytesIO()
            try:
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                print(f"{{IMG_START_TAG}}{{img_str}}{{IMG_END_TAG}}", flush=True)
            except Exception as e:
                print(f"[Remote Graphics Error] {{e}}", flush=True)
            finally:
                plt.close()
                buf.close()

        plt.show = custom_show
        print("🎨 [Remote] Graphics redirection enabled.", flush=True)
    except ImportError:
        pass
"""

    script = f"""
import dill as pickle
import sys
import traceback
import os
import uuid
import base64
import io

# [注入点1：SmartAnnData]
__SMART_CLASS_SOURCE_PLACEHOLDER__

# [注入点2：Graphics Patch]
__GRAPHICS_PATCH_SOURCE_PLACEHOLDER__

RES_START_TAG = "{RES_START_TAG}"
RES_END_TAG = "{RES_END_TAG}"

def execute():
    response = {{'result': None, 'error': None}}
    try:
        patch_matplotlib()

        # === 从 STDIN 读取输入 (包含 AnnData 对象) ===
        input_bytes = sys.stdin.buffer.read()
        if not input_bytes:
            raise ValueError("[Remote] No input received from stdin")
            
        data = pickle.loads(input_bytes)

        func_source = data['func_source']
        func_name = data['func_name']
        args = data['args']
        kwargs = data['kwargs']
        
        local_scope = {{}}
        local_scope['SmartAnnData'] = SmartAnnData
        
        exec(func_source, globals(), local_scope)
        func = local_scope[func_name]
        
        # 仅对显式使用 SmartAnnData 的参数进行加载
        new_args = []
        for arg in args:
            if hasattr(arg, 'load_from_disk'):
                new_args.append(arg.load_from_disk())
            else:
                new_args.append(arg)
        
        new_kwargs = {{}}
        for k, v in kwargs.items():
            if hasattr(v, 'load_from_disk'):
                new_kwargs[k] = v.load_from_disk()
            else:
                new_kwargs[k] = v
        
        if 'cwd' in data:
            if data['cwd'] not in sys.path:
                sys.path.insert(0, data['cwd'])

        print(f"🚀 [Remote] Executing function: {{func_name}} (Memory Mode)", flush=True)
        
        # 执行函数
        result = func(*new_args, **new_kwargs)
        
        final_result_payload = result
        
        # [修改点]：不再强制拦截 AnnData 落盘，直接通过内存返回
        # 除非用户函数内部显式返回了 SmartAnnData 对象
        if hasattr(result, 'save_to_disk') and hasattr(result, 'temp_path'):
             print("💾 [Remote] Result is SmartAnnData, ensuring written to disk...", flush=True)
             saved_path = result.save_to_disk()
             final_result_payload = result # pickle 会序列化这个对象，包含 path
        
        response = {{'result': final_result_payload, 'error_msg': None}}

    except Exception as e:
        tb = traceback.format_exc()
        response = {{'result': None, 'error_msg': str(e), 'traceback': tb}}

    # === 通过 STDOUT 回传结果 (Base64) ===
    try:
        res_bytes = pickle.dumps(response)
        res_b64 = base64.b64encode(res_bytes).decode('utf-8')
        print(f"{{RES_START_TAG}}{{res_b64}}{{RES_END_TAG}}", flush=True)
        
    except Exception as e:
        print(f"Critical error writing result to pipe: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    execute()
"""
    s = script.replace("__SMART_CLASS_SOURCE_PLACEHOLDER__", smart_class_source)
    s = s.replace("__GRAPHICS_PATCH_SOURCE_PLACEHOLDER__", graphics_patch_source)
    return s

# ==========================================
#   3. 装饰器主逻辑
# ==========================================
def wrap(base):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.exists(TEMP_DIR):
                os.makedirs(TEMP_DIR, exist_ok=True)

            smart_wrappers = [] 
            final_result = None
            remote_error = None

            # === 1. 提取源码 ===
            try:
                raw_source = inspect.getsource(func)
                raw_source = textwrap.dedent(raw_source)
                lines = raw_source.split('\n')
                def_line_index = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('def '):
                        def_line_index = i
                        break
                func_source = '\n'.join(lines[def_line_index:])
            except OSError:
                raise RuntimeError("Cannot get function source code.")

            # === 2. 参数处理 (不再强制拦截 AnnData) ===
            new_args = []
            for arg in args:
                # [修改点]：只有显式传入 SmartAnnData 才走磁盘，否则 AnnData 走内存
                if isinstance(arg, SmartAnnData):
                    arg.save_to_disk()
                    new_args.append(arg)
                    smart_wrappers.append(arg)
                else:
                    new_args.append(arg)
            
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, SmartAnnData):
                    v.save_to_disk()
                    new_kwargs[k] = v
                    smart_wrappers.append(v)
                else:
                    new_kwargs[k] = v

            try:
                # === 3. 准备输入包 (内存对象) ===
                data_to_send = {
                    'func_source': func_source, 
                    'func_name': func.__name__, 
                    'args': new_args,
                    'kwargs': new_kwargs,
                    'cwd': os.getcwd() 
                }
                
                input_bytes = pickle.dumps(data_to_send)
                remote_script = _get_remote_script_template()
                
                command = [
                    'conda', 'run', '--no-capture-output', '-n', base,
                    'python', '-u', '-c', remote_script
                ]
                
                print(f"⏳ [Subprocess] Launching '{base}' env...", flush=True)
                
                # === 4. 启动进程与管道交互 ===
                process = subprocess.Popen(
                    command, 
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    bufsize=1
                )

                try:
                    process.stdin.buffer.write(input_bytes)
                    process.stdin.buffer.close() 
                except Exception as e:
                    raise RuntimeError(f"Failed to send data via pipe: {e}")

                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    
                    if line:
                        stripped = line.strip()
                        
                        if IMG_START_TAG in stripped and IMG_END_TAG in stripped:
                            try:
                                b64_data = stripped.split(IMG_START_TAG)[1].split(IMG_END_TAG)[0]
                                img_bytes = base64.b64decode(b64_data)
                                display(IPImage(data=img_bytes))
                            except Exception:
                                print(f"⚠️ [Render Error] Failed to render remote image")
                        
                        elif RES_START_TAG in stripped and RES_END_TAG in stripped:
                            try:
                                res_b64 = stripped.split(RES_START_TAG)[1].split(RES_END_TAG)[0]
                                res_bytes = base64.b64decode(res_b64)
                                result_data = pickle.loads(res_bytes)
                                
                                if result_data.get('error_msg'):
                                    remote_error = result_data
                                else:
                                    final_result = result_data['result']
                            except Exception as e:
                                print(f"⚠️ [Protocol Error] Failed to decode result: {e}")

                        else:
                            print(f"[{base}] {stripped}", flush=True)

                exit_code = process.poll()

                if exit_code != 0:
                    raise RuntimeError(f"Remote process exited with code {exit_code}.")

                if remote_error:
                    print("="*20 + " REMOTE ERROR " + "="*20)
                    print(remote_error.get('traceback', 'No traceback available'))
                    print("="*54)
                    raise RuntimeError(f"Remote execution failed: {remote_error['error_msg']}")

                # === 5. 本地重建结果对象 ===
                # 如果结果是 SmartAnnData (说明在远程被手动包装了)，则加载
                if hasattr(final_result, 'load_from_disk'):
                    final_result = final_result.load_from_disk()
                
                return final_result

            finally:
                for wrapper in smart_wrappers:
                    wrapper.cleanup()

        return wrapper
    return decorator