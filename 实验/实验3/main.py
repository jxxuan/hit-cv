import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import tqdm
import sift


def getFiles(train, path):
    images = []
    for file in os.listdir(path):
        images.append(os.path.join(path, file))
    if train is True:
        np.random.shuffle(images)
    return images


# def getDescriptors(sift, img):
#     kp, des = sift.detectAndCompute(img, None)
#     return des

def getDescriptors(img):
    kp, des = sift.computeKeypointsAndDescriptors(img)
    return des


def readImage(img_path):
    img = cv2.imread(img_path, 0)
    return cv2.resize(img, (150, 150))


def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in tqdm.tqdm(descriptor_list[1:]):
        descriptors = np.vstack((descriptors, descriptor))
    return descriptors


def clusterDescriptors(descriptors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters).fit(descriptors)
    return kmeans


def extractFeatures(kmeans, descriptor_list, image_count, num_clusters):
    im_features = np.array([np.zeros(num_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1
    return im_features


def normalizeFeatures(scale, features):
    return scale.transform(features)


def plotHistogram(im_features, num_clusters):
    x_scalar = np.arange(num_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:, h], dtype=np.int32)) for h in range(num_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()


def svcParamSelection(X, y, kernel, nfolds):
    Cs = [0.5, 0.1, 0.15, 0.2, 0.3]
    gammas = [0.1, 0.11, 0.095, 0.105]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_


def findSVM(im_features, train_labels):
    features = im_features

    # params = svcParamSelection(features, train_labels, kernel, 5)
    # C_param, gamma_param = params.get("C"), params.get("gamma")
    # print(C_param, gamma_param)
    # class_weight = {
    #     0: (807 / (7 * 140)),
    #     1: (807 / (7 * 140)),
    #     2: (807 / (7 * 133)),
    #     3: (807 / (7 * 70)),
    #     4: (807 / (7 * 42)),
    #     5: (807 / (7 * 140)),
    #     6: (807 / (7 * 142))
    # }

    # svm = SVC(kernel=kernel, C=C_param, gamma=gamma_param, class_weight=class_weight)
    print("Fitting SVM.")
    svm = SVC(C=15.0)
    svm.fit(features, train_labels)

    return svm


def plotConfusionMatrix(y_true, y_pred, classes,
                        normalize=False,
                        title=None,
                        cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plotConfusions(true, predictions):
    np.set_printoptions(precision=2)

    class_names = ["cat", "wolf", "cheetah", "chimpanzee", "hamster"]
    plotConfusionMatrix(true, predictions, classes=class_names,
                        title='Confusion matrix, without normalization')

    plotConfusionMatrix(true, predictions, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')

    plt.show()


def findAccuracy(true, predictions):
    print('accuracy score: %0.3f' % accuracy_score(true, predictions))


def trainModel(num_clusters):
    path = 'raw_image/training'
    images = getFiles(True, path)
    print("Train images path detected.")
    # sift = cv2.xfeatures2d.SIFT_create()
    descriptor_list = []
    train_labels = np.array([])
    image_count = 0

    for img_path in tqdm.tqdm(images):
        name = os.path.basename(img_path)
        class_index = int(name.split('_')[0])
        img = cv2.imread(img_path, 0)
        # des = getDescriptors(sift, img)
        des = getDescriptors(img)
        if des is not None and class_index in [0, 2, 4, 6, 8]:
            class_index = class_index // 2
            descriptor_list.append(des)
            train_labels = np.append(train_labels, class_index)
            image_count += 1
        else:
            images.remove(img_path)

    print(image_count, 'images accepted.')

    descriptors = vstackDescriptors(descriptor_list)
    print("Descriptors vstacked.")

    kmeans = clusterDescriptors(descriptors, num_clusters)
    print("Descriptors clustered.")

    im_features = extractFeatures(kmeans, descriptor_list, image_count, num_clusters)
    print("Images features extracted.")

    scale = StandardScaler().fit(im_features)
    im_features = scale.transform(im_features)
    print("Train images normalized.")

    plotHistogram(im_features, num_clusters)
    print("Features histogram plotted.")

    svm = findSVM(im_features, train_labels)
    print("SVM fitted.")
    print("Training completed.")

    return kmeans, scale, svm


def testModel(kmeans, scale, svm, num_clusters):
    path = 'raw_image/testing'
    test_images = getFiles(False, path)
    print("Test images path detected.")

    image_count = 0
    true = []
    descriptor_list = []

    name_dict = {
        0: "cat",
        1: "wolf",
        2: "cheetah",
        3: "chimpanzee",
        4: "hamster",
    }

    sift = cv2.xfeatures2d.SIFT_create()

    for img_path in test_images:
        img = readImage(img_path)
        des = getDescriptors(sift, img)
        name = os.path.basename(img_path)
        class_index = int(name.split('_')[0])

        if des is not None and class_index in [0, 2, 4, 6, 8]:
            class_index = class_index // 2
            image_count += 1
            descriptor_list.append(des)
            class_name = name_dict[class_index]
            true.append(class_name)

    print(image_count, 'images accepted.')

    test_features = extractFeatures(kmeans, descriptor_list, image_count, num_clusters)

    test_features = scale.transform(test_features)

    kernel_test = test_features

    predictions = [name_dict[int(i)] for i in svm.predict(kernel_test)]
    print("Test images classified.")

    plotConfusions(true, predictions)
    print("Confusion matrixes plotted.")


    findAccuracy(true, predictions)
    print("Accuracy calculated.")
    print("Execution done.")


if __name__ == '__main__':
    num_clusters = 100
    kmeans, scale, svm= trainModel(num_clusters)
    testModel(kmeans, scale, svm, num_clusters)
