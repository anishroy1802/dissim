% Define the e4 function in MATLAB
function result = e4(x)
    % Your implementation of the e4 function here
    % Compute the objective value and return it as a scalar
    result = -(multinodal(x(1)) + multinodal(x(2)));
end

% Define the multinodal function in MATLAB (if not defined elsewhere)
function y = multinodal(x)
    y = (sin(0.05 * pi * x).^6) ./ (2.^(2 * ((x - 10) / 80).^2));
end
