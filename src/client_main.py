import random
from datetime import datetime
import argparse
from main_study import *
import os


def main_function(stu_list, fw):
    fw.write("-----------------stu_list-----------------\n")
    fw.write(str(stu_list) + '\n' + '\n')

    win = visual.Window(size=MIN_SIZE, fullscr=False, screen=1, winType='pyglet', allowGUI=False, allowStencil=False,
                        monitor='testMonitor', color=[-1, -1, -1], colorSpace='rgb', blendMode='avg', useFBO=True,
                        units='pix')

    instruction_train = visual.TextStim(win, text='请仔细阅读实验说明,\n按下空格开始实验训练', pos=(0, 0),
                                        height=WORD_HEIGHT,
                                        color=(1, 1, 1), alignText='center')
    instruction_end = visual.TextStim(win, text='实验结束,请联系主试', pos=(0, TEXT_HEIGHT), height=WORD_HEIGHT,
                                      color=(1, 1, 1),
                                      alignText='center')
    instruction_main = visual.TextStim(win, text='按空格键,开始正式实验', pos=(0, TEXT_HEIGHT), height=WORD_HEIGHT,
                                       color=(1, 1, 1), alignText='center')

    # 1 准备训练
    # instruction_train.draw()
    # win.flip()

    # event.waitKeys(keyList=['space'])

    # 1 训练
    # point_study(win, stu_list[:1], fw)

    # 1 准备正式实验
    instruction_main.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

    # 1 正式实验
    main_study(win, stu_list, fw)

    # end
    instruction_end.draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    win.close()
    core.quit()
    fw.close()


def load_question_info(category_name_file):
    total_arr = []
    categoty_dic = {}
    with open(category_name_file, encoding='GBK') as f:
        for line in f.readlines():
            line_list = line.strip().split()
            categoty_dic[line_list[0]] = line_list[1]
    for dir_path, dir_names, file_names in os.walk('./data/imageNet_images/'):
        if len(dir_names) == 0:
            img_dic = {"category": categoty_dic[dir_path[-9:]], 'image': []}
            for file_name in file_names:
                if '.JPEG' in file_name:
                    img_dic['image'].append(os.path.join(dir_path, file_name))
            random.shuffle(img_dic['image'])
            total_arr.append(img_dic)
    # 随机化所有类别的顺序
    random.shuffle(total_arr)
    return total_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect image-EEG data in same category paradigm.')
    parser.add_argument('-s', '--seed', type=int, default=54321, help='random seed')
    args = parser.parse_args()
    random.seed(args.seed)

    total_arr = load_question_info('./data/imageNet_images/synset_map_ch.txt')

    main_function(total_arr,
                  open(os.path.join('./record/', datetime.now().strftime('%Y%m%d') + '_' + str(args.seed)), 'w',
                       encoding='utf8'))
