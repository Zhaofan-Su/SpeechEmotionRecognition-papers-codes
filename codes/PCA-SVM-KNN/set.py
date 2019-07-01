import os

root = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(root, 'svm')
path = os.path.join(base, 'database')

dirs = os.listdir(path)
for d in dirs:
    if d == 'wav':
        wp = os.path.join(path, d)
        files = os.listdir(wp)
        for file in files:
            if file.find('N') != -1:
                # 中性
                os.rename(os.path.join(wp, file),
                          os.path.join(os.path.join(path, 'neutral'), file))
            elif file.find('W') != -1:
                # 愤怒
                os.rename(os.path.join(wp, file),
                          os.path.join(os.path.join(path, 'angry'), file))
            elif file.find('T') != -1:
                # 悲伤
                os.rename(os.path.join(wp, file),
                          os.path.join(os.path.join(path, 'sad'), file))
            elif file.find('F') != -1:
                # 高兴
                os.rename(os.path.join(wp, file),
                          os.path.join(os.path.join(path, 'happy'), file))
            elif file.find('A') != -1:
                # 害怕
                os.rename(os.path.join(wp, file),
                          os.path.join(os.path.join(path, 'fear'), file))
            elif file.find('E') != -1:
                # 惊讶
                os.rename(os.path.join(wp, file),
                          os.path.join(os.path.join(path, 'neutral'), file))