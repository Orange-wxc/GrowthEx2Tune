import socket
import time
import globalValue
from maEnv import datautils

class Server():
    def __init__(self, event):
        self.event = event

    def server(self):
        print('server thread start....')
        # status数组
        globalValue.GLOBAL_STATUS = [0] * 5

        #create socket
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM,0)

        #bind
        # s.bind(('192.168.2.30',20000))
        # s.bind(('222.20.74.140',20000))
        s.bind(('172.16.56.237', 20000))


        #listen
        s.listen(1)
        print('Waiting for connection......')

        #多线程连接5
        #def tcplink(sock,addr):
        #    print('Accept new connection from %s:%s...' % addr)
        #   while True:
        #       data = sock.recv(1024)
        #       time.sleep(1)
        #       if not data or data.decode('utf-8') == 'exit':
        #           break
        #       print('recvied data:',data.decode('utf-8'))
        #       sock.send(('Hello000.%s!' % data.decode('utf-8')).encode('utf-8'))
        #   sock.close()
        #   print('connection from %s:%s closed' % addr)



        #handle connection
        while True:
            sock,addr = s.accept()
            #print(sock)
            ##print(addr)
            #print('Accept new connection from %s:%s...' % addr)

            while True:
                # 接收数据库的状态信息
                reseived_data = sock.recv(1024)
                time.sleep(1)

                # close the connection
                if not reseived_data or reseived_data.decode('utf-8').startswith('exit'):
                    break

                # 处理接收到的数据并返回推荐值给数据库端
                # 解码
                reseived_data = reseived_data.decode('utf-8')
                print(reseived_data)
                # 处理收到的监控数据
                state = datautils.GetState(reseived_data)
                print(state)

                print('Waiting for action...')
                self.event.wait()
                send_len = sock.send(globalValue.GLOBAL_ACTION.encode('utf-8'))
                print('globalValue.GLOBAL_ACTION',globalValue.GLOBAL_ACTION)
                # self.event.clear()
                # 发送参数推荐配置给数据库端
                # send_len = sock.send('okokokokokokokok'.encode('utf-8'))
                # print("send_len = ",send_len)

                # state = GetState(reseived_data)
                # state = list(map(int,state))

                #更新GLOBAL_STATUS
                # globalValue.GLOBAL_STATUS[globalValue.GLOBAL_STATUS_POSITION] = state
                # globalValue.GLOBAL_STATUS_POSITION = (globalValue.GLOBAL_STATUS_POSITION + 1) % globalValue.GLOBAL_STATUS_LIMIT
                # globalValue.GLOBAL_FREE_LEN = state[0]

                #print(globalValue.GLOBAL_STATUS)
                #if(globalValue.GLOBAL_STATUS_POSITION == globalValue.GLOBAL_STATUS_LIMIT-1):
                #    break
                #print('state:',state)

            sock.close()
            #print('connection from %s:%s closed' % addr)
            #break

    # 对mysqld端发送的监控数据进行处理
    # 数据分为如下及部分
    # 1.负载表示       load[]
    # 2.可调整参数     knobs[]
    # 3.性能参数      performance[]



