# -*- coding: utf-8 -*-
"""
МОДУЛЬ ВЫЧИСЛЕНИЙ — автоматический выбор бэкенда (CPU/GPU/распределённый)

Поддерживает:
1. NumPy (CPU) — по умолчанию
2. CuPy (GPU) — если доступен
3. JAX (GPU/TPU) — экспериментально
4. Распределённый режим (Ray/Dask) — для кластера

Этот модуль позволяет масштабировать систему от ПК до суперкомпьютера.
"""

import os
import sys
import numpy as np

# --- ГЛОБАЛЬНЫЙ БЭКЕНД ---
_BACKEND = None
_DEVICE = None
_IS_GPU_AVAILABLE = False


def get_backend():
    """Возвращает текущий вычислительный бэкенд."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = detect_backend()
    return _BACKEND


def detect_backend(force_backend: str = None) -> str:
    """
    Автоматически определяет лучший доступный бэкенд.
    
    Приоритет: CUDA (CuPy) > CPU (NumPy) > JAX (если доступен)
    """
    global _IS_GPU_AVAILABLE
    
    if force_backend:
        if force_backend == 'cupy':
            try:
                import cupy as cp
                _IS_GPU_AVAILABLE = True
                print(f"✅ GPU бэкенд (CuPy) активирован принудительно")
                return 'cupy'
            except ImportError:
                print(f"⚠️ CuPy не установлен, используем NumPy")
                return 'numpy'
        elif force_backend == 'jax':
            try:
                import jax
                print(f"✅ JAX бэкенд активирован принудительно")
                return 'jax'
            except ImportError:
                print(f"⚠️ JAX не установлен, используем NumPy")
                return 'numpy'
        else:
            return 'numpy'
    
    # 1. Пробуем CuPy (GPU)
    try:
        import cupy as cp
        # Проверяем, что CUDA доступна
        cp.cuda.runtime.getDeviceCount()
        _IS_GPU_AVAILABLE = True
        print(f"✅ GPU бэкенд (CuPy) обнаружен")
        return 'cupy'
    except (ImportError, RuntimeError):
        pass
    
    # 2. Пробуем JAX (GPU/TPU)
    try:
        import jax
        import jax.numpy as jnp
        # Проверяем, что GPU доступен
        if jax.devices('gpu'):
            print(f"✅ JAX бэкенд (GPU) обнаружен")
            return 'jax'
        elif jax.devices('tpu'):
            print(f"✅ JAX бэкенд (TPU) обнаружен")
            return 'jax'
    except (ImportError, RuntimeError):
        pass
    
    # 3. По умолчанию — NumPy (CPU)
    print(f"ℹ️ Используем CPU бэкенд (NumPy)")
    return 'numpy'


def get_xp():
    """
    Возвращает модуль для вычислений (cupy, jax.numpy или numpy).
    Используйте этот модуль для всех операций, чтобы код работал и на CPU, и на GPU.
    """
    backend = get_backend()
    
    if backend == 'cupy':
        import cupy as cp
        return cp
    elif backend == 'jax':
        import jax.numpy as jnp
        return jnp
    else:
        return np


def get_device():
    """Возвращает текущее устройство (CPU/GPU)."""
    backend = get_backend()
    if backend == 'cupy':
        import cupy as cp
        return cp.cuda.Device(0)
    elif backend == 'jax':
        import jax
        devices = jax.devices('gpu')
        if devices:
            return devices[0]
        return 'cpu'
    else:
        return 'cpu'


def to_device(array):
    """
    Переносит массив на текущее устройство (CPU/GPU).
    """
    backend = get_backend()
    xp = get_xp()
    
    if backend == 'cupy':
        import cupy as cp
        if isinstance(array, np.ndarray):
            return cp.asarray(array)
        return array
    elif backend == 'jax':
        import jax.numpy as jnp
        if isinstance(array, np.ndarray):
            return jnp.array(array)
        return array
    else:
        if hasattr(array, 'get'):
            return array.get()
        return array


def to_numpy(array):
    """
    Преобразует массив в NumPy (для совместимости с другими библиотеками).
    """
    backend = get_backend()
    
    if backend == 'cupy':
        import cupy as cp
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array
    elif backend == 'jax':
        import jax.numpy as jnp
        if isinstance(array, jnp.ndarray):
            return np.array(array)
        return array
    else:
        return array


def is_gpu_available():
    """Проверяет, доступен ли GPU."""
    global _IS_GPU_AVAILABLE
    if _IS_GPU_AVAILABLE is None:
        detect_backend()
    return _IS_GPU_AVAILABLE


def get_backend_name():
    """Возвращает имя текущего бэкенда."""
    return get_backend()


# --- РАСПРЕДЕЛЁННЫЕ ВЫЧИСЛЕНИЯ (Ray) ---
_RAY_AVAILABLE = False
_RAY_INITIALIZED = False


def init_ray(address: str = None, num_cpus: int = None, num_gpus: int = None):
    """
    Инициализирует Ray для распределённых вычислений.
    
    Args:
        address: адрес кластера (None для локального режима)
        num_cpus: количество CPU
        num_gpus: количество GPU
    """
    global _RAY_AVAILABLE, _RAY_INITIALIZED
    
    try:
        import ray
        if not ray.is_initialized():
            ray.init(address=address, num_cpus=num_cpus, num_gpus=num_gpus)
        _RAY_AVAILABLE = True
        _RAY_INITIALIZED = True
        print(f"✅ Ray инициализирован (адрес: {address or 'локальный'})")
        return True
    except ImportError:
        print(f"⚠️ Ray не установлен. Установите: pip install ray")
        return False
    except Exception as e:
        print(f"❌ Ошибка инициализации Ray: {e}")
        return False


def is_ray_available():
    """Проверяет, доступен ли Ray."""
    global _RAY_AVAILABLE
    return _RAY_AVAILABLE


def get_cluster_resources():
    """Возвращает информацию о ресурсах кластера."""
    if not is_ray_available():
        return {'status': 'ray_not_available'}
    
    try:
        import ray
        if not ray.is_initialized():
            return {'status': 'not_initialized'}
        
        resources = ray.cluster_resources()
        return {
            'status': 'ok',
            'cpus': resources.get('CPU', 0),
            'gpus': resources.get('GPU', 0),
            'memory': resources.get('memory', 0),
            'object_store_memory': resources.get('object_store_memory', 0),
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def is_cupy_available():
    """Проверяет, доступен ли CuPy."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except:
        return False


def is_jax_available():
    """Проверяет, доступен ли JAX."""
    try:
        import jax
        return True
    except:
        return False


# --- НАСТРОЙКА ПО УМОЛЧАНИЮ ---
# Автоматически определяем бэкенд при импорте
_BACKEND = detect_backend()

print(f"🔧 Вычислительный бэкенд: {_BACKEND}")
print(f"   GPU доступен: {is_gpu_available()}")
if is_gpu_available():
    print(f"   Устройство: {get_device()}")
