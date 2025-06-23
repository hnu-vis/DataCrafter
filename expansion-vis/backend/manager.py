from flask_script import Manager, Server
from application import create_app

app = create_app()
manager = Manager(app)

# 添加Server命令，覆盖默认的5000端口
manager.add_command("run", Server(port=8081))

@manager.command
def run(port=8082):
    app.run(host="0.0.0.0", port=port, debug=True)

if __name__ == "__main__":
    manager.run()
