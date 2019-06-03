import sys
sys.path.append('/home/antonio/caffe-master/python')

import caffe
import numpy as np
import cv2
print(caffe)

#this function is to get the location of bird which is the high conv response area
def bird_box(net):
	mask = net.blobs['upsample'].data.reshape(448,448)
	mask_max = np.max(mask.flat)
	t = mask_max * 0.1
	t1 = np.max(mask,axis = 0)
	for j in range(448):
		if t1[j]>t:
			left = j
			break
	for j in range(447,-1,-1):
		if t1[j]>t:
			right = j
			break
	t2 = np.max(mask,axis = 1)
	for j in range(448):
		if t2[j]>t:
			up = j
			break
	for j in range(447,-1,-1):
		if t2[j]>t:
			down = j
			break
	x = (left + right)/2
	y = (up + down)/2
	l = np.maximum(right - left, down - up)/2
	return x,y,l

#this function is to get four part locations which is the output of our designed part network
def part_box(net):
	part1 = np.argmax(net.blobs['mask_1_4'].data.reshape(28,28))
	part2 = np.argmax(net.blobs['mask_2_4'].data.reshape(28,28))
	part3 = np.argmax(net.blobs['mask_3_4'].data.reshape(28,28))
	part4 = np.argmax(net.blobs['mask_4_4'].data.reshape(28,28))
	return part1 % 28, part1 / 28, part2 % 28, part2 / 28, part3 % 28, part3 / 28, part4 % 28, part4 / 28

#input the original image and bird location, crop the bird area and resize to 224*224
def get_bird(img,x,y,l):
	[n,m,_]=img.shape
	if m>n:
		islong=1
	else:
		islong=0
	if islong==0:
		x =  x*m/448
		y =  y*m/448 + (n - m) / 2
		l =  l*m/448
	else:
		x =  x*n/448 + (m - n) / 2
		y =  y*n/448
		l =  l*n/448
	box = (np.maximum(0,np.int(x - l)), np.maximum(0,np.int(y - l)),
		np.minimum(m,np.int(x + l)), np.minimum(n,np.int(y + l)))
	return cv2.resize(img[box[1]:box[3],box[0]:box[2],:],(224,224))

#input the original image and part locations, crop the four parts and resize to 224*224
def get_part(img,parts):
	img_parts = [[] for i in range(4)]
	[n,m,_]=img.shape
	if m > n:
		islong = 1
	else:
		islong = 0
	for i in range(4):
		if islong == 0: # m < n
			parts[i][0] = parts[i][0]*m/28+8
			parts[i][1] = parts[i][1]*m/28+8 + (n - m) / 2
			l = 64*m/448
		else: 
			parts[i][0] = parts[i][0]*n/28+8 + (m - n) / 2
			parts[i][1] = parts[i][1]*n/28+8
			l = 64*n/448
		# print(parts[i], l)
		box = (np.maximum(0,np.int(parts[i][0] - l)), np.maximum(0,np.int(parts[i][1] - l)),
		 np.minimum(m,np.int(parts[i][0] + l)), np.minimum(n,np.int(parts[i][1] + l)))
		img_parts[i] = cv2.resize(img[box[1]:box[3],box[0]:box[2],:],(224,224))
	return img_parts

#translate an img to fit the input of a network
def data_trans(img, shape):
	mu = np.array([109.973,127.338,123.883])
	transformer = caffe.io.Transformer({'data': shape})
	transformer.set_transpose('data', (2,0,1))  
	transformer.set_mean('data', mu)			
	transformer.set_raw_scale('data', 255)	 
	# transformer.set_channel_swap('data', (2,1,0)) 
	transformed_image = transformer.preprocess('data', img)
	return transformed_image

#crop the centor part from 448*n (keep image ratio) to 448*448
def crop_centor(img):
	[n,m,_]=img.shape
	if m > n:
		m = int(m*448/n)
		n = 448
	else:
		n = int(n*448/m)
		m = 448
	resized = cv2.resize(img,(m,n))/255.0
	trans = data_trans(resized, (1,3,n,m))
	return trans[:,int((n-448)/2):int((n+448)/2),int((m-448)/2):int((m+448)/2)]


#crop the centor part from 256*n (keep image ratio) to 224*224
def crop_lit_centor(img):
	[n,m,_]=img.shape
	if m>n:
		m = int(m*256/n)
		n = 256
	else:
		n = int(n*256/m)
		m=256
	return data_trans(cv2.resize(img,(m,n))/255.0,(1,3,n,m))[:,int((n-224)/2):int((n+224)/2),int((m-224)/2):int((m+224)/2)]


ls = 5794
# caffe.set_mode_gpu()
# caffe.set_device(12)
# mu = array([109.973,127.338,123.883])
model_weights ='model/bird_part.caffemodel'
model_def ='deploy/bird_part_deploy.prototxt'
p_net = caffe.Net(model_def,model_weights,caffe.TEST) # part network

model_weights ='model/bird_class.caffemodel'
model_def ='deploy/bird_cls_deploy.prototxt'
c_net = caffe.Net(model_def,model_weights,caffe.TEST) # classification network

test_list = open('bird_data/test_list.txt').readlines()

done_list = open('result/result.txt').readlines()
def done_func(entry):
	# index species place p1 p2 p3 p4 p5
	parts = entry.split(' ')
	return int(parts[0]), (parts[1], int(parts[2]))
	
done_map = dict(map(lambda entry: done_func(entry), done_list))

def append_done(index, species, place, p1, p2, p3, p4, p5, fileName):
	line = '%d %s %d %.3f %.3f %.3f %.3f %.3f %s\n' % (index, species, place, p1, p2, p3, p4, p5, fileName)
	f = open('result/result.txt', 'a+')
	f.write(line)
	f.close()

do_output_images = 0
do_networking = 1
counter = 0
total = 0
count = 1
do_output_weights = 0

def save_attention(data, centerImage, name):
	
	cv2.imwrite('/home/antonio/1_cv macnn/'+name+'.png', data)

accuracy = 0
for i in range(ls):	
	if i>-1:
		if i in done_map.keys():
			
			if done_map[i][1] == 0:
				accuracy += 1 
				
			total += 1
			
		else:
			
			v0 = test_list[i].split(' ')[0]
			v1 = v0.split('\\')
			fileName = v1[1]
			species = fileName.split('_0')[0]
			v2 = v1[0] + '.' + species + '/' + fileName;
			img_path = '/home/antonio/CUB_200_2011/images/' + v2
			
			# print(img_path)
			
			img = cv2.imread(img_path)
			if img.ndim < 3:
				img = np.transpose(np.array([img,img,img]),(1,2,0))
			
			[n,m,_] = img.shape
			label = np.int(test_list[i].split(' ')[1])
			center_cropped = crop_centor(img) # only the center is evaluated
			
			lit_data = crop_lit_centor(cv2.resize(img,(256,256))) # stretched
			p_net.blobs['data'].data[...] = center_cropped
			p_out = p_net.forward()
			x,y,l = bird_box(p_net)
			part_boxs = part_box(p_net)
			
			if do_output_weights != 0:
				save_attention(img, 0, 'original')
				save_attention(cv2.resize(img,(256,256)), 0, 'lit')
				save_attention(p_net.blobs['attention'].data.reshape(28,28) / 255, 0, 'attention')
				save_attention(p_net.blobs['mask_1_4'].data.reshape(28,28) * 255, 0, 'm1')
				save_attention(p_net.blobs['mask_2_4'].data.reshape(28,28) * 255, 0, 'm2')
				save_attention(p_net.blobs['mask_3_4'].data.reshape(28,28) * 255, 0, 'm3')
				save_attention(p_net.blobs['mask_4_4'].data.reshape(28,28) * 255, 0, 'm4')
			
			
			# print('boxes:')
			# print(part_boxs)
			img_bird = get_bird(img,x,y,l)
			part_boxes = np.array(part_boxs).reshape((4,2))
			img_part = get_part(img, part_boxes)
			# print('parts:')
			# print(img_part)
			# print(reshaped_part_boxes)
			
			if do_output_weights != 0:
				save_attention(img_bird, 0, 'bird_box')
			
			c_net.blobs['label'].data[...] = label # to check whether it's correct
			c_net.blobs['ori_data'].data[...] = lit_data # stretched, center part 224 x 224 of 256 x 256
			c_net.blobs['bird_data'].data[...] = data_trans(img_bird/255.0,(1,3,224,224))
			c_net.blobs['part1_data'].data[...] = data_trans(img_part[0]/255.0,(1,3,224,224))
			c_net.blobs['part2_data'].data[...] = data_trans(img_part[1]/255.0,(1,3,224,224))
			c_net.blobs['part3_data'].data[...] = data_trans(img_part[2]/255.0,(1,3,224,224))
			c_net.blobs['part4_data'].data[...] = data_trans(img_part[3]/255.0,(1,3,224,224))
			
			
			if do_networking != 0:
				c_out = c_net.forward()
				
				# pins = c_net.blobs['fc'].data
				# print(pins)
				top_x = c_net.blobs['accuracy_top_x'].data.flatten()
				
				# find the best 5 values...
				top_5 = np.sort(top_x)
				
				if c_net.blobs['accuracy'].data < 1.0:
					place = 199 - np.where(top_5 == top_x[label])[0][-1] # starting at 0 = best
				else:
					place = 0
				
				append_done(i, species, place, top_5[199], top_5[198], top_5[197], top_5[196], top_5[195], fileName)
				
				# pin_string = [item for item in pins.astype(str)]
				# print(label, pin_string)
				
				accuracy += c_net.blobs['accuracy'].data
				total += 1
				
				print(fileName, accuracy / total)
				
				if do_output_images != 0:
					l = 64 * min(n,m) / 448
					for rect in reshaped_part_boxes:
						cv2.rectangle(img,(int(rect[0]-l),int(rect[1]-l)),(int(rect[0]+l),int(rect[1]+l)),(0,255,0),1)
						
					cv2.imwrite('/home/antonio/1_cv macnn/'+str(i)+'.'+str(int(c_net.blobs['accuracy'].data))+'.png', img)
				
				counter += 1
				if counter >= count:
					break
				
			else:
				
				l = 64 * min(n,m) / 448
				for rect in reshaped_part_boxes:
					cv2.rectangle(img,(int(rect[0]-l),int(rect[1]-l)),(int(rect[0]+l),int(rect[1]+l)),(0,255,0),1)
					
				cv2.imshow('Result', img)
				cv2.waitKey()
			
				break

# erste 50: 82%

if do_networking != 0:
	print('accuracy:', accuracy / total)


