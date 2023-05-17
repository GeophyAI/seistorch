import sys, select
import subprocess
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget, QFileDialog


class WorkerThread(QThread):
    script_finished = pyqtSignal(str)  # 自定义信号

    def __init__(self):
        super().__init__()
        self.script_queue = []

    def add_script(self, script, script_name):
        self.script_queue.append((script, script_name))

    def run(self):
        for script, script_name in self.script_queue:
            try:
                # 执行命令并捕获输出
                process = subprocess.Popen(script, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                
                # 读取命令输出流的数据
                while True:
                    # 检查命令输出流是否有可读数据
                    if process.stdout in select.select([process.stdout], [], [], 0)[0]:
                        # 读取命令输出流的一行数据
                        line = process.stdout.readline().strip()
                        if line:
                            print(line)
                    
                    # 检查命令是否执行完成
                    if process.poll() is not None:
                        break

                if process.returncode != 0:
                    print(f"命令执行失败，返回码：{process.returncode}")
            except Exception as e:
                print(f"命令执行出错：{str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.script_path_text_edit = QTextEdit(self)
        self.result_text_edit = QTextEdit(self)
        self.run_button = QPushButton("运行", self)
        self.run_button.clicked.connect(self.add_script_to_queue)

        self.toolbar = self.addToolBar("工具栏")
        self.toolbar.addWidget(self.run_button)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.script_path_text_edit)
        self.layout.addWidget(self.result_text_edit)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

        self.worker_thread = WorkerThread()
        self.worker_thread.script_finished.connect(self.on_script_finished)

    def add_script_to_queue(self):
        script = self.load_script()
        if script:
            script_name = f"脚本 {self.script_path_text_edit.toPlainText()}\n"
            self.worker_thread.add_script(script, script_name)
            self.script_path_text_edit.append(f"添加脚本：{script_name}\n")
            self.script_path_text_edit.append("----------------------")

            if not self.worker_thread.isRunning():
                self.worker_thread.start()

    def on_script_finished(self, result):
        self.result_text_edit.append(result)
        self.result_text_edit.append("----------------------")

    def load_script(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Sh files (*.sh)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.script_path_text_edit.append(f"加载脚本：{file_path}\n")
            with open(file_path, "r") as file:
                script = file.read()
            return script

        return None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
