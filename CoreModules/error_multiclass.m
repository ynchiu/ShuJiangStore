% -------------------------------------------------------------------------
function err = error_multiclass(labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
if length(size(predictions))==2
    predictions=permute(predictions,[3,4,1,2]);
end
[~,predictions] = sort(predictions, 3, 'descend') ;


% be resilient to badly formatted labels
if numel(labels) == size(predictions, 4)
  labels = reshape(labels,1,1,1,[]) ;
end

% skip null labels
mass = single(labels(:,:,1,:) > 0) ;
if size(labels,3) == 2
  % if there is a second channel in labels, used it as weights
  mass = mass .* labels(:,:,2,:) ;
  labels(:,:,2,:) = [] ;
end

error = ~bsxfun(@eq, predictions, labels) ;
error=gather(error);
err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:5,:),[],3)))) ;