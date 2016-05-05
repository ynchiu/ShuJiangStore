function y = tanh_ln(x,dzdy)

    if nargin <= 1 || isempty(dzdy)
        y = (exp(x)-exp(-x)) ./ (exp(x)+exp(-x));
    else
        y = 4./(exp(x)+exp(-x)).^2;
    end

end