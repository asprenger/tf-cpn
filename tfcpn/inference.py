from collections import namedtuple
import cv2
import numpy as np
import tensorflow as tf
from tfcpn.network import Network
from tfcpn.dataset import Preprocessing

def nms(dets, thresh):
    """
    Simple NMS implementations. Python version translated from Cython method `cpu_nms` at:
    https://github.com/asprenger/tf-cpn/blob/master/lib/lib_kernel/lib_nms/nms/cpu_nms.pyx

    Compares each pair of bboxes. If a bbox pair iou > threshold only the bbox with the
    higher score is kept.

    Args:
        dets: bbox [x1, y1, x2, y2, score]
        thresh: iou threshold 
    Returns:
        keep: array with bbox indices that are kept
    """
    assert dets.shape[1]==5
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    # nominal indices
    # _i, _j
    
    # sorted indices
    # i, j
    
    # temp variables for box i's (the box currently under consideration)
    # ix1, iy, ix2, iy2, iarea
    
    # variables for computing overlap with box j (lower scoring box)
    # xx1, yy1, xx2, yy2, w, h, inter, ovr

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep

def x1y1x2y2_to_x1y1wh(bbox):
    """Convert [x1,y1,x2,y2] bounding box to a [x,y,w,h] 
    bounding box"""
    assert len(bbox)==4
    assert bbox[0] <= bbox[2] and bbox[1] <= bbox[3]
    return [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]


TFCpnConfig = namedtuple('TFCpnConfig', ['nr_skeleton', 'symmetry', 'data_shape', 'output_shape', 'pixel_means'])

class CPNPoseEstimator(object):

    def __init__(self, checkpoint_path, tf_config=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(config=tf_config)
            with self.sess.as_default():
                # build model
                net = Network()
                net.make_network(is_train=False)
                # restore checkpoint
                saver = tf.train.Saver()
                sess = tf.Session()
                saver.restore(sess, checkpoint_path)
                images = net._inputs[0]
                output = net._outputs[0]
                self.sess = sess
                self.images = images
                self.output = output 

        cfg = TFCpnConfig(
            nr_skeleton = 17,
            symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)],
            data_shape = (images.shape[1].value, images.shape[2].value), # height, width
            output_shape = (output.shape[1].value, output.shape[2].value), # height, width
            pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]]) # BGR
        )
        self.cfg = cfg


    def inference(self, image, bboxes, bbox_scores):

        # estimate poses
        test_data = []
        for bbox, score in zip(bboxes, bbox_scores):
            test_data.append({
                'bbox': x1y1x2y2_to_x1y1wh(bbox),
                'score': score
            })

        # TODO understand and add to signature    
        nms_thresh = 1.0
        min_scores = 1e-10
        min_box_size = 0.

        cfg = self.cfg

        # cls_dets contains bboxs and scores: [x1,y1,x2,y2, score]
        cls_dets = np.zeros((len(test_data), 5), dtype=np.float32)
        for i in range(len(test_data)):
            bbox = np.asarray(test_data[i]['bbox'])
            cls_dets[i, :4] = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) # xywh => x1y1x2y2
            cls_dets[i, 4] = np.array(test_data[i]['score'])

        # filter bboxes that fail the `min_scores` and `min_box_size` thresholds
        keep = np.where((cls_dets[:, 4] >= min_scores) &
                        ((cls_dets[:, 3] - cls_dets[:, 1]) * (cls_dets[:, 2] - cls_dets[:, 0]) >= min_box_size))[0]
        cls_dets = cls_dets[keep]

        # filter redundant boxes by NMS
        keep = nms(cls_dets, nms_thresh)
        cls_dets = cls_dets[keep]



        cls_skeleton = np.zeros((len(test_data), cfg.nr_skeleton, 3)) # kpt_x, kpt_y, score
        crops = np.zeros((len(test_data), 4))

        test_imgs = [] # list of (1, 3, 256, 192) np.float32, normalized, CHW, BGR
        details = []
        for i in range(len(test_data)):
            test_img, detail = Preprocessing(test_data[i], stage='test', image=image) 
            test_imgs.append(test_img)
            details.append(detail)

        details = np.asarray(details)
        feed = test_imgs

        # flip each image vertically and append to `feed`
        for i in range(len(test_data)):
            ori_img = test_imgs[i][0].transpose(1, 2, 0)
            flip_img = cv2.flip(ori_img, 1)
            feed.append(flip_img.transpose(2, 0, 1)[np.newaxis, ...])

        feed = np.vstack(feed) # (nb_images*2, 3, 256, 192)
        feed_dict = {
            self.images: feed.transpose(0, 2, 3, 1).astype(np.float32)
        }
        res = self.sess.run(self.output, feed_dict=feed_dict) # (nb_images*2, 64, 48, 17)
        res = res.transpose(0, 3, 1, 2) # (nb_images*2, 17, 64, 48)

        # average the heatmaps of the original and the flipped image
        # to get the final heatmap
        for i in range(len(test_data)):
            # select heatmap of a flipped image and flip it    
            fmp = res[len(test_data) + i].transpose((1, 2, 0))
            fmp = cv2.flip(fmp, 1)
            fmp = list(fmp.transpose((2, 0, 1)))
            # switch left/right side keypoints maps
            for (q, w) in cfg.symmetry:
                fmp[q], fmp[w] = fmp[w], fmp[q]
            fmp = np.array(fmp)
            # average the heatmaps
            res[i] += fmp
            res[i] /= 2

        # now the first half of `res` contains the averaged maps, the second half 
        # contains the maps of the flipped images and is not needed any more.

        for test_image_id in range(len(test_data)):

            # copy res and rescale approx. to [0.5, 1.5]
            # r0 will be used to calculate keypoint scores
            r0 = res[test_image_id].copy()        
            r0 /= 255.
            r0 += 0.5

            # rescale `res` so that max==1.0
            for w in range(cfg.nr_skeleton):
                res[test_image_id, w] /= np.amax(res[test_image_id, w])

            # `dr` has the size of `res` with some border on each side
            # copy `res` in the middle of `dr`
            # not sure why we add the border, maybe because of the bluring
            border = 10
            dr = np.zeros((cfg.nr_skeleton, cfg.output_shape[0] + 2 * border, cfg.output_shape[1] + 2 * border)) # (17, 84, 68)
            dr[:, border:-border, border:-border] = res[test_image_id][:cfg.nr_skeleton].copy()

            # blur dr
            for w in range(cfg.nr_skeleton):
                dr[w] = cv2.GaussianBlur(dr[w], (21, 21), 0)    

            for w in range(cfg.nr_skeleton):
                # locate the point with the max value in each blured map
                lb = dr[w].argmax()
                y, x = np.unravel_index(lb, dr[w].shape)

                # now locate the point with the second highest max value in each map
                dr[w, y, x] = 0
                lb = dr[w].argmax()
                py, px = np.unravel_index(lb, dr[w].shape)

                # calculate the distance between the two max points.
                # the border is subtracted, because the coordinates are determined 
                # based on `dr` that has been padded with the boarder. this translates
                # the coordinates back into the `res` matrix
                y -= border
                x -= border
                py -= border + y
                px -= border + x        
                ln = (px ** 2 + py ** 2) ** 0.5

                # a quarter offset in the direction from the highest response to the 
                # second highest response is used to obtain the final location of 
                # the keypoints.
                delta = 0.25
                if ln > 1e-3:
                    # TODO: check this, this does not look right!
                    x += delta * px / ln
                    y += delta * py / ln

                # make sure x,y are not out of bounds        
                x = max(0, min(x, cfg.output_shape[1] - 1))
                y = max(0, min(y, cfg.output_shape[0] - 1))

                # x,y are based on the (64, 48) output shape, multiply by 4
                # to scale them back to the size of the input image (256, 192)
                cls_skeleton[test_image_id, w, :2] = (x * 4 + 2, y * 4 + 2)

                # use the value at (x,y) in `r0` as score for the keypoint
                cls_skeleton[test_image_id, w, 2] = r0[w, int(round(y) + 1e-10), int(round(x) + 1e-10)]

            # the x,y coordinates in `cls_skeleton` are located on the (256, 192) input 
            # image, now map them back to the original image 
            crops[test_image_id, :] = details[test_image_id, :]  
            for w in range(cfg.nr_skeleton):
                cls_skeleton[test_image_id, w, 0] = cls_skeleton[test_image_id, w, 0] / cfg.data_shape[1] * (crops[test_image_id][2] - crops[test_image_id][0]) + crops[test_image_id][0]
                cls_skeleton[test_image_id, w, 1] = cls_skeleton[test_image_id, w, 1] / cfg.data_shape[0] * (crops[test_image_id][3] - crops[test_image_id][1]) + crops[test_image_id][1]

        # Rescoring: the product of boxes’ score and the average score of all keypoints 
        # is considered as the final pose score of a person instance.    

        cls_scores = cls_dets[:, -1].copy() # box scores (3,)
        cls_partsco = cls_skeleton[:, :, 2].copy() # keypoint scores (nb_bboxes, 17)      
        cls_dets[:, -1] = cls_scores * cls_partsco.mean(axis=1)

        # cls_dets # bbox x1y1x2y2 and score, shape: (nb_boxes, 5)
        # cls_skeleton # keypoints x,y and score, shape: (nb_boxes,17,3)
        
        bboxes_with_score = [[int(x[0]), int(x[1]), int(x[2]), int(x[3]), x[4]] for x in cls_dets]
        poses = [[[int(kpt[0]), int(kpt[1]), kpt[2]] for kpt in pose] for pose in cls_skeleton]

        return bboxes_with_score, poses

