import csv
import numpy


def amp_to_int(amp):
    if amp == 'A':
        return 0
    elif amp == 'M':
        return 1
    elif amp == 'P':
        return 2
    else:
        return -1


# CLS ###############################################################################

# train classes

train_cls_data = []
with open('data/GCM_Training.cls') as train_cls:
    reader = csv.reader(train_cls, delimiter=' ')
    for row in reader:
        train_cls_data.append(row)

class_count = int(train_cls_data[0][1])
class_desc = train_cls_data[1][1:]

train_classes = train_cls_data[2]
train_classes_binary = numpy.zeros(shape=(len(train_classes), class_count), dtype=int)

for i in range(len(train_classes)):
    train_classes_binary[i, train_classes[i]] = 1

print('class count ', class_count)
print('class desc ', class_desc)
print()
print()

print('train classes ', train_classes)
print('train classes binary ', train_classes_binary)
print()


# test classes

test_cls_data = []
with open('data/GCM_Test.cls') as test_cls:
    reader = csv.reader(test_cls, delimiter=' ')
    for row in reader:
        test_cls_data.append(row)

# remove Met class - not present in paper or train dataset
test_classes = test_cls_data[2][0:46]
test_classes_binary = numpy.zeros(shape=(len(test_classes), class_count), dtype=int)

for i in range(len(test_classes)):
    test_classes_binary[i, test_classes[i]] = 1

print('test classes ', test_classes)
print('test classes binary ', test_classes_binary)
print()


# RES ###############################################################################

# train data

train_values = numpy.zeros(shape=(144, 16063))
train_amp = numpy.zeros(shape=(144, 16063, 3), dtype=int)
with open('data/GCM_Training.res') as train_res:
    reader = csv.reader(train_res, delimiter='\t')
    j = 0
    for row in reader:
        if j > 2:
            for i in range(144):
                train_values[i, j - 3] = row[2 * i + 2]
                train_amp[i, j - 3, amp_to_int(row[2 * i + 3])] = 1
        j += 1

print('train values ', train_values)
print('train amp ', train_amp)
print()


# test data

test_values = numpy.zeros(shape=(46, 16063))
test_amp = numpy.zeros(shape=(46, 16063, 3), dtype=int)
with open('data/GCM_Test.res') as test_res:
    reader = csv.reader(test_res, delimiter='\t')
    j = 0
    for row in reader:
        if j > 2:
            for i in range(46):
                test_values[i, j - 3] = row[2 * i + 2]
                test_amp[i, j - 3, amp_to_int(row[2 * i + 3])] = 1
        j += 1

print('test values ', test_values)
print('test amp ', test_amp)
print()