import os
import sys
import getopt
from importlib import import_module

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
LESSONS_DIR = os.path.join(BASE_DIR, 'lessons')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
OPTIONS = 'g:l:a'
LONG_OPTIONS = ['lesson', 'all']


def open_lessons():
    for root, dirs, files in os.walk(LESSONS_DIR):
        for d in dirs:
            open_lesson(d)


def open_lesson(lesson):
    lesson_root = os.path.join(LESSONS_DIR, str(lesson))
    for root, dirs, files in os.walk(lesson_root):
        if 'main.py' in files:
            print('----------------------------------')
            print('Lesson #{} running'.format(lesson))
            print('-')
            module_name = 'lessons.{}.main'.format(lesson)
            m = import_module(module_name, package=__name__)
            m.run()
            print('----------------------------------')


def generate_lesson(lesson):
    lesson_root = os.path.join(LESSONS_DIR, str(lesson))
    if os.path.exists(lesson_root):
        raise Exception('Lesson with #{} already exists!'.format(lesson))
    os.mkdir(lesson_root)
    with open(os.path.join(lesson_root, '__init__.py'), 'a') as new_file:
        new_file.writelines('__all__ = [\'run\']\n')
    with open(os.path.join(lesson_root, 'main.py'), 'w') as new_file:
        with open(os.path.join(TEMPLATES_DIR, 'main.py.lesson.txt'), 'r') as template:
            new_file.writelines(template.readlines())
    print('----------------------------------')
    print('Lesson #{} was successfully generated.'.format(lesson))
    print('----------------------------------')


def main(argv):
    try:
        opts, args = getopt.getopt(argv, OPTIONS, LONG_OPTIONS)
    except getopt.GetoptError:
        print('Argument parser failed.')
        sys.exit(2)

    for option, value in opts:
        if option == '-g' and value:
            generate_lesson(value)
            return
        elif option == '-l' and value:
            open_lesson(value)
            return
        elif option == '-a':
            open_lessons()
            return

    raise Exception('One argument should be passed!')


if __name__ == '__main__':
    main(sys.argv[1:])
