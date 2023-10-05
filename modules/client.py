import requests

session = requests.Session()
session.trust_env = False
url = 'http://localhost:6745'


def send_model_save_callback(path):
    resp = session.post(url + '/callback/loreSaved', json={'path': path})


class TrainStatus:
    def __init__(self):
        self.epoch = 0
        self.step = 0
        self.total_epoch = 0
        self.total_step = 0
        self.current_loss = 0
        self.avg_loss =0
    def output(self):
        return {
            'epoch': self.epoch,
            'step': self.step,
            'total_epoch': self.total_epoch,
            'total_step': self.total_step,
        }


def update_train_status(status):
    # try:
    #     request = grequests.post(url + '/callback/trainProgress', json=status)
    #     response = grequests.map([request])[0]
    # except:
    #     pass
    resp = session.post(url + '/callback/trainProgress', json={'status': status.output()})


def send_train_status(status):
    update_train_status(status)
