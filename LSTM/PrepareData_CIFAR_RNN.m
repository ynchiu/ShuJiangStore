function [ opts ] = PrepareData_CIFAR_RNN( opts )


imdb=getCifarImdb(opts);

opts.train=double(imdb.images.data(:,:,:,imdb.images.set==1));
norm_factor=max(abs(opts.train(:)));
opts.train=opts.train./norm_factor;
opts.train=permute(opts.train,[1,3,2,4]);
opts.n_train=size(opts.train,3)*size(opts.train,4);
opts.train=reshape(opts.train,[size(opts.train,1),size(opts.train,2),opts.n_train]);
opts.train=permute(opts.train,[2,3,1]);
opts.test=double(imdb.images.data(:,:,:,imdb.images.set==3))./norm_factor;
opts.test=permute(opts.test,[1,3,2,4]);
opts.n_test=size(opts.test,3)*size(opts.test,4);
opts.test=reshape(opts.test,[size(opts.test,1),size(opts.test,2),opts.n_test]);
opts.test=permute(opts.test,[2,3,1]);

end

