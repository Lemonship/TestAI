wmic Path win32_process Where "name = 'python.exe' and CommandLine Like '%%tensorboard.exe%%'" Call Terminate

start chrome "http://localhost:6006"
tensorboard --logdir=log
