path = 'F:\temp\';
fid = fopen('temp.txt', 'rb');
a = fread(fid, inf, 'float');
b = reshape(a, 128, []);
b = b';

dim = 6075;
for i = 1:128
	img = b((i-1)*6075+1:(i-1)*6075+6075);
	img = reshape(img, 45,45,3);
	imwrite(img/255, [path, num2str(i), '.jpg']);
end