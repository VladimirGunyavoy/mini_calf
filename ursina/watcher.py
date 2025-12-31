"""
Вотчер для автоматической перезагрузки приложения
Перезапускает приложение ТОЛЬКО когда оно закрыто пользователем (exit code 0)
НЕ отслеживает изменения файлов - полностью убрана вся логика отслеживания

Использование:
    python watcher.py                  # запускает main.py (по умолчанию)
    python watcher.py train_calf_visual.py  # запускает указанный скрипт
"""

import time
import os
import subprocess
import sys

# --- НАСТРОЙКИ ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Папка ursina
PYTHON_EXECUTABLE = sys.executable
DEFAULT_SCRIPT = "main.py"
# --- КОНЕЦ НАСТРОЕК ---

class ScriptRunner:
    def __init__(self, script_name=None):
        self.process = None
        self.script_name = script_name or DEFAULT_SCRIPT
        self.script_path = os.path.join(SCRIPT_DIR, self.script_name)
        self.running = True
        self.restart_delay = 1.0
        
    def run_script(self):
        """Запуск скрипта"""
        if not os.path.exists(self.script_path):
            print(f"[WATCHER] Ошибка: Скрипт {self.script_name} не найден")
            return False
            
        try:
            if self.process and self.process.poll() is None:
                print("[WATCHER] Останавливаю предыдущий процесс...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print("[WATCHER] Принудительное завершение процесса...")
                    self.process.kill()
                    self.process.wait()
            
            print(f"[WATCHER] Запуск скрипта: {self.script_name}")
            self.process = subprocess.Popen(
                [PYTHON_EXECUTABLE, "-u", self.script_path],
                cwd=SCRIPT_DIR
            )
            print(f"[WATCHER] Скрипт запущен, PID: {self.process.pid}")
            return True
            
        except Exception as e:
            print(f"[WATCHER] Ошибка запуска скрипта: {e}")
            return False
    
    def check_and_restart(self):
        """Проверка состояния процесса - перезапуск ТОЛЬКО при закрытии пользователем"""
        if self.process and self.process.poll() is not None:
            exit_code = self.process.returncode
            if exit_code == 0:
                # Приложение закрыто пользователем - перезапускаем
                print(f"[WATCHER] Приложение закрыто пользователем (код {exit_code}), перезапуск...")
                time.sleep(self.restart_delay)
                self.run_script()
            else:
                # Ошибка - не перезапускаем автоматически
                print(f"[WATCHER] Приложение завершилось с ошибкой (код {exit_code})")
                print("[WATCHER] Ожидаю ручного перезапуска...")
                self.running = False
    
    def stop(self):
        """Остановить все процессы"""
        self.running = False
        if self.process and self.process.poll() is None:
            print("[WATCHER] Остановка процесса...")
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()

def main():
    """Основная функция - только перезапуск при закрытии игры"""
    # Получаем имя скрипта из аргументов командной строки
    script_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCRIPT
    
    print("=" * 60)
    print("WATCHER - Перезапуск только при закрытии игры")
    print("=" * 60)
    print(f"Скрипт: {script_name}")
    print("=" * 60)
    print("Вотчер НЕ отслеживает изменения файлов")
    print("Перезапуск только когда вы закрываете игру (Q/Escape)")
    print("Нажмите Ctrl+C для остановки вотчера")
    print()
    
    runner = ScriptRunner(script_name)
    
    # Запускаем скрипт первый раз
    if not runner.run_script():
        print("\n[WATCHER] Не удалось запустить скрипт. Завершение работы.")
        sys.exit(1)
    
    try:
        # Просто ждем завершения процесса и перезапускаем при exit code 0
        while runner.running:
            runner.check_and_restart()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[WATCHER] Получен сигнал прерывания, завершаем работу...")
    finally:
        runner.stop()
        print("[WATCHER] Программа завершена")

if __name__ == "__main__":
    main()
