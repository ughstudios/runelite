import psutil
import time

def find_runelite_processes():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'RuneLite' in proc.info['name'] or 'java' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if cmdline:
                    print(cmdline)
                if cmdline and any('RuneLite' in arg for arg in cmdline):
                    print(f"\nProcess: {proc.info['name']} (PID: {proc.info['pid']})")
                    print("Command line arguments:")
                    for arg in cmdline:
                        print(f"  {arg}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

print("Monitoring for RuneLite processes. Launch RuneLite through the Jagex Launcher now.")
print("Press Ctrl+C to stop monitoring.\n")

try:
    while True:
        find_runelite_processes()
        time.sleep(2)
except KeyboardInterrupt:
    print("\nStopped monitoring.") 