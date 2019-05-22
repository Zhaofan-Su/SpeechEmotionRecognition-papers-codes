import argparse
from Base import processing, train_svm, predict


def cmd():

    option = 'p'
    database = 'casia'
    radar = True
    model_name = 'default'
    label_num = '6'
    model_type = 'svm'

    paser = argparse.ArgumentParser(description='Speech Emotion Recognition')

    paser.add_argument(
        '-o',
        '--option',
        type=str,
        dest='option',
        help='Use p to predict directly and t to train your model')
    paser.add_argument(
        '-db', '--database', type=str, dest='database', help='Your database')
    paser.add_argument(
        '-r',
        '--radar',
        type=int,
        dest='radar',
        help='Whether to use the radar image or not')
    paser.add_argument(
        '-m', '--model', type=str, dest='model', help='The name of your model')
    paser.add_argument(
        '-l',
        '--labels',
        type=int,
        dest='labels',
        help='The number of labels you use')

    args = paser.parse_args()

    option = args.option.lower()
    database = args.database if args.database else 'casia'
    radar = args.radar if args.radar else True
    model_name = args.model if args.model else 'default'
    label_num = str(args.labels) if args.labels else '6'

    if option == 'p':
        # 预测
        predict(database, model_name, label_num, radar, model_type)
    elif option == 't':
        # 训练
        # 预处理保存音频数据的特征
        features = processing(database, label_num, model_type)
        train_svm(database, features, label_num, model_name, model_type)
    else:
        print("Please choose the right option: p for predict, t for train")
        return


if __name__ == '__main__':
    cmd()
