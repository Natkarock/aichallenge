import sys, time, threading
class Spinner:
    def __init__(self,text='Минутку, подбираю идеи'):
        self.text=text
        self.stop_flag=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True)
    def _run(self):
        frames=['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']; i=0
        while not self.stop_flag.is_set():
            sys.stdout.write(f'\r{frames[i%len(frames)]} {self.text}…'); sys.stdout.flush(); i+=1; time.sleep(0.08)
    def start(self): self.thread.start()
    def stop_and_clear(self): self.stop_flag.set(); self.thread.join(timeout=1); sys.stdout.write('\r\033[2K'); sys.stdout.flush()
