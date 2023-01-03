%% 2D Black-Scholes PDE
% Zachary Humphries
% COMP 521
% Fall 2022

clear
close all

%% Parameters

strike = 1;                 % Strike Price
T = 1;                      % Simulation time or Final Maturity Time

a = 0;                      % Minimum Value of Option for Asset X (must be zero)
b = round(10*strike);       % Maximum Value of Option for Asset X per recommendation of reference paper (between 8*K and 12*K)
c = 0;                      % Minimum Value of Option for Asset Y (must be zero)
d = b;                      % Maximum Value of Option for Asset X

m = 8* round(10*strike);    % Personal Preference: Gives Enough Divisions for a More Accurate Result
n = m;                      % Number of cells along the y-axis

dx = (b-a)/m;               % Step length along the x-axis
dy = (d-c)/n;               % Step length along the y-axis


dt = 0.001;                 % Personal Preference: Much less than Von Neumann stability criterion for explicit scheme dx^2/(4) (about 0.0039)

omega11 = 0.3;              % Omega_xx = Omega_yy of the volatility correlation matrix
omega12 = 0.05;             % Omega_xy = Omega_yx of the volatility correlation matrix
r = 0.1;                    % Risk free interest rate

%% Setting Up Matricies of F, X, and Y

xgrid = [a : dx :  b];
ygrid = [c : dy :  d];

[X, Y] = meshgrid(xgrid, ygrid);

Xmatrix = diag(reshape(X, (m+1)*(n+1), 1));            % Diagonal Matrix of X Mesh for Calculating A
Ymatrix = diag(reshape(Y, (m+1)*(n+1), 1));            % Diagonal Matrix of Y Mesh for Calculating A

%% Setting Up Matrix for Fx

Fx = Fx_Matrix(m,n,dx,dy);                          % 2nd Order 2D Scheme for First Derivative with Respect to X
Fy = Fy_Matrix(m,n,dx,dy);                          % 2nd Order 2D Scheme for First Derivative with Respect to Y

Fxx = Fxx_Matrix(m,n,dx,dy);                        % 2nd Order 2D Scheme for Second Derivative with Respect to X
Fyy = Fyy_Matrix(m,n,dx,dy);                        % 2nd Order 2D Scheme for Second Derivative with Respect to Y

Fxy = Fxy_Matrix(m,n,dx,dy);                        % 2nd Order 2D Scheme for Mixed Derivative with Respect to X and Y

sub_matrix = diag(diag(comp_matrix("x", m, n)));    % Matrix to Subtract from speye so All Boundary Conditions in A are Zero

%% Using Black-Scholes PDE to Create A (Excluding Boundary Conditions)

A = (-r*Xmatrix*Fx) - (r*Ymatrix*Fy) - ((1/2)*omega11^2*Xmatrix*Xmatrix * Fxx) - ((1/2)*omega11^2*Ymatrix*Ymatrix * Fyy) - (omega12^2*Xmatrix*Ymatrix*Fxy) + (r*(speye((m+1)*(n+1))-sub_matrix));

%% Encorporating Close-Field Boundary Conditions into A

Fx_1D = Derivative_1D_Matrix(m,dx);                 % 2nd Order 1D Scheme for First Derivative with Respect to X
Fy_1D = Derivative_1D_Matrix(n,dy);                 % 2nd Order 1D Scheme for First Derivative with Respect to Y
Fxx_1D = Double_Derivative_1D_Matrix(m,dx);         % 2nd Order 1D Scheme for Second Derivative with Respect to X
Fyy_1D = Double_Derivative_1D_Matrix(n,dy);         % 2nd Order 1D Scheme for Second Derivative with Respect to Y

Xmatrix_1D = diag(xgrid');
Ymatrix_1D = diag(ygrid');

I_1D = speye(m+1,n+1);                              % Origin and Far-Field Boundary Conditions Are Later Addressed
I_1D(end,end) = 0;
I_1D(1,1) = 0;

xaxis = ((-r * Xmatrix_1D*Fx_1D) - (1/2 * omega11^2 * Xmatrix_1D*Xmatrix_1D * Fxx_1D) + r*I_1D);
yaxis = ((-r * Ymatrix_1D*Fy_1D) - (1/2 * omega11^2 * Ymatrix_1D*Ymatrix_1D * Fyy_1D) + r*I_1D);


A(1:m+1, 1:n+1) = sparse(xaxis);                    % Inserting Close-Field Boundary Condition for X-Axis into A

row_insert = [1:m+1:(m+1)*(n+1)];                   % Resizing Y to be Inserted Into A Matrix
yaxis_matrix1 = sparse((m+1)*(n+1),m+1);
yaxis_matrix1(row_insert,:) = yaxis;
col_insert = [1:n+1:(n+1)*(m+1)];
yaxis_matrix2 = sparse((m+1)*(n+1),(m+1)*(n+1));
yaxis_matrix2(:, col_insert) = yaxis_matrix1;

A = A+yaxis_matrix2;                                % Inserting Close-Field Boundary Condition for Y-Axis into A


%% Updating A to Account for Far-Field Dirichlet Boundary Conditions

dirichlet_far = zeros((m+1),(n+1));
dirichlet_far(end,:) = ones(length(xgrid),1);
dirichlet_far(:,end) = ones(length(ygrid),1);

dirichlet_far = diag(reshape(dirichlet_far, 1, (m+1)*(n+1)));

A = sparse(A+dirichlet_far);                    % Values Corresponding to Far-Field Boundary in A Are One on Diagonal
A(1,1) = 1;                                     % Origin is Always Zero

%% Creating Far-Field Dirichlet Boundary Condition Values

BC = zeros(m+1, n+1);

uppery = ((b+ygrid)/2)-(strike*exp(-r*(0)));    % Updating Boundary Conditions
upperx = ((xgrid+d)/2)-(strike*exp(-r*(0)));    % Updating Boundary Conditions
BC(end,:) = upperx;
BC(:,end) = uppery;

BC = reshape(BC,(m+1)*(n+1), 1);

%% Initial Values for time = T

ICV = max(((X+Y)/2)-strike, 0);

%% 1st Order Time Scheme to Calculate U After First Time Step

U = reshape(ICV, (m+1)*(n+1), 1);

U_minus = U;
BC_minus = BC;

U = inv((speye(size(A))+(dt*A)))*U_minus+(dt*BC_minus);

U = U-BC;

BC = reshape(BC,(m+1),(n+1));
upperx = ((xgrid+d)/2)-(strike*exp(-r*(T)));    % Updating Far-Field Boundary Conditions for X
uppery = ((b+ygrid)/2)-(strike*exp(-r*(T)));    % Updating Far-Field Boundary Conditions for Y

BC(end,:) = upperx;
BC(:,end) = uppery;
BC = reshape(BC,(m+1)*(n+1), 1);

U = reshape(U,(m+1),(n+1));
U(end,:) = 0;
U(:,end) = 0;
U = reshape(U,(m+1)*(n+1), 1);

U = U + BC;                                     % Making Sure Far-Field Boundaries Have Correct Value in Case of Rounding Error

%% Calculate Inverse of Matrix Needed for 2nd Order Implicit Time Scheme

A_second_order = inv(A+((1/(2*dt))*speye(size(A))));

%% Time Integration Loop
% Note: Value is Being Discounted back to the Present from Exersize Date

count = 1;
len = length(dt : dt : T)-1;

for t = dt : dt : T-dt 
    fprintf("%f ",count)
    fprintf("%f \n",len)

    count = count + 1;

    top = ((((U-BC))/dt)+((dt/2)*((-2*(U-BC) + (U_minus-BC_minus))/(dt*dt))) + BC); % Updating Implicit Scheme Vector

    top = reshape(top,(m+1),(n+1));                 % Making Sure Far-Field Boundaries Have Correct Value in Case of Rounding Error
    top(end,:) = 0;
    top(:,end) = 0;
    top = reshape(top,(m+1)*(n+1), 1);
    top = top + BC;

    U_plus = A_second_order*top;                    % 2nd Order Implicit Scheme for Next Time Step

    BC_minus = BC;

    BC = reshape(BC,(m+1),(n+1));
    upperx = ((xgrid+d)/2)-(strike*exp(-r*(T-t)));  % Updating Far-Field Boundary Conditions for X
    uppery = ((b+ygrid)/2)-(strike*exp(-r*(T-t)));  % Updating Far-Field Boundary Conditions for Y

    BC(end,:) = upperx;
    BC(:,end) = uppery;
    BC = reshape(BC,(m+1)*(n+1), 1);

    U_plus = reshape(U_plus,(m+1),(n+1));
    U_plus(end,:) = 0;
    U_plus(:,end) = 0;
    U_plus = reshape(U_plus,(m+1)*(n+1), 1);
    
    U_plus = U_plus + BC;

    U_minus = U;                                    % Updating U_minus and U for Next Time Step
    U = U_plus;

end

%% Graphing Final U

U_graph = max(U_plus, 0);                           % Option Value doesn't go below zero
surf(X, Y, reshape(U_graph, m+1, n+1))
title(['2D Black-Scholes \newlineTime = ' num2str(T-t-dt, '%1.4f')])
xlabel('x')
ylabel('y')
zlabel('F')
colorbar
caxis([-0.01 strike])
axis([0 5*strike/3 0 5*strike/3 -0.01 strike])      % Examining Boundary Up to 5*strike/3 as Done in Paper

% caxis([-0.05, (b+d-strike)/2])                    % Uncomment for Graph of All of U
% axis([a b c d -0.05 (b+d-strike)/2])
drawnow

%% Error Testing
mult_list = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64];
value_list = error_testing(mult_list, strike, T, dx, dy, omega11, omega12, r);

value_list_adj = abs((value_list(1:(end-1))-value_list(end))');

dt_list = dt*mult_list(1:(end-1));

%% Error plotting
figure
forward_poly = polyfit(log(dt_list), log(value_list_adj),1);
loglog(dt_list, value_list_adj, "o-"); grid on;
title("Implicit Forward Euler Error")
subtitle_name_forward = strcat("$\log(Appox Error_{x=1.25,y=1.25}) = ", sprintf("%2.6f", forward_poly(1)), "\log(\Delta t) + ", sprintf("%2.6f", forward_poly(2)), "$");
subtitle(subtitle_name_forward,'interpreter','latex')
xlabel("$\log(\Delta t)$",'interpreter','latex')
ylabel("$\log(Appox Error)$",'interpreter','latex')

%% Functions

function matrixdx = Derivative_1D_Matrix(m,dx)
    one = ones(m+1,1);
    sparse_m = sparse(m+1,1);
    A = spdiags([-1*one sparse_m one],-1:1,m+1,m+1);
    A(1,:) = sparse_m';
    A(end,:) = sparse_m';
 
    matrixdx = (A)/(2*dx);
end

function matrixdxx = Double_Derivative_1D_Matrix(m,dx)
    one = ones(m+1,1);
    sparse_m = sparse(m+1,1);
    A = spdiags([one -2*one one],-1:1,m+1,m+1);
    A(1,:) = sparse_m';
    A(end,:) = sparse_m';

    matrixdxx = (A)/(dx^2);
end

function matrixdx = Fx_Matrix(m,n,dx,dy)
    one = ones(m+1,1);
    sparse_m = sparse(m+1,1);
    A = spdiags([-1*one sparse_m one],-1:1,m+1,m+1);
    A(1,:) = sparse_m';
    A(end,:) = sparse_m';

    sparse_y = speye(n+1,n+1);
    sparse_n = sparse(n+1,1);
    sparse_y(:,1) = sparse_n;
    sparse_y(:,end) = sparse_n;
 
    matrixdx = kron(sparse_y, A)/(2*dx);
end

function matrixdy = Fy_Matrix(m,n,dx,dy)
    one = ones(n-1,1);
    sparse_n = sparse(n+1,1);
    one = sparse([0;one;0]);
    one_list = repmat(one,m-1,1);
    one_list1 = [sparse_n; one_list; sparse_n];
    one_list2 = [sparse_n; one_list; sparse_n];

    A = spdiags([-1*one_list1 repmat(sparse_n, n+1, 2*(m+1)-1) one_list2],-(m+1):(m+1),(m+1)*(n+1),(m+1)*(n+1));
 
    matrixdy = -1*(((A)/(2*dy))');
end

function matrixdxx = Fxx_Matrix(m,n,dx,dy)
    one = ones(m+1,1);
    sparse_m = sparse(m+1,1);
    A = spdiags([one -2*one one],-1:1,m+1,m+1);
    A(1,:) = sparse_m';
    A(end,:) = sparse_m';

    sparse_y = speye(n+1,n+1);
    sparse_n = sparse(n+1,1);
    sparse_y(:,1) = sparse_n;
    sparse_y(:,end) = sparse_n;
 
    matrixdxx = kron(sparse_y, A)/(dx^2);
end

function matrixdyy = Fyy_Matrix(m,n,dx,dy)
    one = ones(n-1,1);
    sparse_n = sparse(n+1,1);
    one = sparse([0;one;0]);
    one_list = repmat(one,m-1,1);
    one_list1 = [sparse_n; one_list; sparse_n];
    one_list2 = [sparse_n; one_list; sparse_n];
    diag = [sparse_n; one_list; sparse_n];
    
    A = spdiags([one_list1 repmat(sparse_n, n+1, m) -2*diag repmat(sparse_n, n+1, m) one_list2],-(m+1):(m+1),(m+1)*(n+1),(m+1)*(n+1));
 
    matrixdyy = ((A)/(dy^2))';
end

function matrixdxy = Fxy_Matrix(m,n,dx,dy)
    one = ones(n-1,1);
    sparse_n = sparse(n+1,1);
    one1 = sparse([0;one;0]);
    one2 = sparse([0;one;0]);
    one_list = repmat(one1,m-1,1);
    one_list2 = repmat(one2,m-1,1);
    one_list1 = [sparse_n; one_list; sparse_n];
    one_list2 = [sparse_n; one_list2; sparse_n];
    one_list3 = [sparse_n; one_list2; sparse_n];
    one_list4 = [sparse_n; one_list; sparse_n];

    sparse_list = repmat(sparse_n,m+1,1);

    diags1 = [one_list1 sparse_list -1*one_list2];

    diags2 = [one_list1 sparse_list -1*one_list2];


    A1 = spdiags(diags1,-(m+1)-1:-(m+1)+1,(m+1)*(n+1),(m+1)*(n+1));

    A2 = spdiags(diags2,(m+1)-1:(m+1)+1,(m+1)*(n+1),(m+1)*(n+1));

    A = A1+A2;

    matrixdxy = ((A)/(4*dy*dy))';
end

function A = comp_matrix(yee, m, n)
    bc_matrix = sparse(m+1,n+1);
    bc_matrix(1,:) = 1;
    bc_matrix(end,:) = 1;
    bc_matrix(:,1) = 1;
    bc_matrix(:,end) = 1;
    
    bc_list = reshape(bc_matrix, (m+1)*(n+1),1);
    bc_matrix = repmat(bc_list, 1, (m+1)*(n+1));

    if yee=="x"
        A = bc_matrix;
    else
        A = bc_matrix';
    end
end

function error_list = error_testing(mult_list, strike, T, dx, dy, omega11, omega12, r)
    error_list = zeros(length(mult_list),1);
    for ii = [1:length(mult_list)]
        mult = mult_list(ii);
        %% Spatial discretization

        a = 0;                      % Minimum Value of Option for Asset X (must be zero)
        b = round(10*strike);       % Maximum Value of Option for Asset X
        c = 0;                      % Minimum Value of Option for Asset Y (must be zero)
        d = b;                      % Maximum Value of Option for Asset X
        
        m = 8* round(10*strike);    % Personal Preference: Gives Enough Divisions for a More Accurate Result
        n = m;                      % Number of cells along the y-axis
        
        dx = (b-a)/m;               % Step length along the x-axis
        dy = (d-c)/n;               % Step length along the y-axis

        dt = 0.001 * mult;
               
        %% Setting Up Matricies of F, X, and Y
        
        xgrid = [a : dx :  b];
        ygrid = [c : dy :  d];
        
        [X, Y] = meshgrid(xgrid, ygrid);
        
        Xmatrix = diag(reshape(X, (m+1)*(n+1), 1));            % Diagonal Matrix of X Mesh for Calculating A
        Ymatrix = diag(reshape(Y, (m+1)*(n+1), 1));            % Diagonal Matrix of Y Mesh for Calculating A
        
        %% Setting Up Matrix for Fx
        
        Fx = Fx_Matrix(m,n,dx,dy);                          % 2nd Order 2D Scheme for First Derivative with Respect to X
        Fy = Fy_Matrix(m,n,dx,dy);                          % 2nd Order 2D Scheme for First Derivative with Respect to Y
        
        Fxx = Fxx_Matrix(m,n,dx,dy);                        % 2nd Order 2D Scheme for Second Derivative with Respect to X
        Fyy = Fyy_Matrix(m,n,dx,dy);                        % 2nd Order 2D Scheme for Second Derivative with Respect to Y
        
        Fxy = Fxy_Matrix(m,n,dx,dy);                        % 2nd Order 2D Scheme for Mixed Derivative with Respect to X and Y
        
        sub_matrix = diag(diag(comp_matrix("x", m, n)));    % Matrix to Subtract from speye so All Boundary Conditions in A are Zero
        
        %% Using Black-Scholes PDE to Create A (Excluding Boundary Conditions)
        
        A = (-r*Xmatrix*Fx) - (r*Ymatrix*Fy) - ((1/2)*omega11^2*Xmatrix*Xmatrix * Fxx) - ((1/2)*omega11^2*Ymatrix*Ymatrix * Fyy) - (omega12^2*Xmatrix*Ymatrix*Fxy) + (r*(speye((m+1)*(n+1))-sub_matrix));
        
        %% Encorporating Close-Field Boundary Conditions into A
        
        Fx_1D = Derivative_1D_Matrix(m,dx);                 % 2nd Order 1D Scheme for First Derivative with Respect to X
        Fy_1D = Derivative_1D_Matrix(n,dy);                 % 2nd Order 1D Scheme for First Derivative with Respect to Y
        Fxx_1D = Double_Derivative_1D_Matrix(m,dx);         % 2nd Order 1D Scheme for Second Derivative with Respect to X
        Fyy_1D = Double_Derivative_1D_Matrix(n,dy);         % 2nd Order 1D Scheme for Second Derivative with Respect to Y
        
        Xmatrix_1D = diag(xgrid');
        Ymatrix_1D = diag(ygrid');
        
        I_1D = speye(m+1,n+1);                              % Origin and Far-Field Boundary Conditions Are Later Addressed
        I_1D(end,end) = 0;
        I_1D(1,1) = 0;
        
        xaxis = ((-r * Xmatrix_1D*Fx_1D) - (1/2 * omega11^2 * Xmatrix_1D*Xmatrix_1D * Fxx_1D) + r*I_1D);
        yaxis = ((-r * Ymatrix_1D*Fy_1D) - (1/2 * omega11^2 * Ymatrix_1D*Ymatrix_1D * Fyy_1D) + r*I_1D);
        
        
        A(1:m+1, 1:n+1) = sparse(xaxis);                    % Inserting Close-Field Boundary Condition for X-Axis into A
        
        row_insert = [1:m+1:(m+1)*(n+1)];                   % Resizing Y to be Inserted Into A Matrix
        yaxis_matrix1 = sparse((m+1)*(n+1),m+1);
        yaxis_matrix1(row_insert,:) = yaxis;
        col_insert = [1:n+1:(n+1)*(m+1)];
        yaxis_matrix2 = sparse((m+1)*(n+1),(m+1)*(n+1));
        yaxis_matrix2(:, col_insert) = yaxis_matrix1;
        
        A = A+yaxis_matrix2;                                % Inserting Close-Field Boundary Condition for Y-Axis into A
        
        
        %% Updating A to Account for Far-Field Dirichlet Boundary Conditions
        
        dirichlet_far = zeros((m+1),(n+1));
        dirichlet_far(end,:) = ones(length(xgrid),1);
        dirichlet_far(:,end) = ones(length(ygrid),1);
        
        dirichlet_far = diag(reshape(dirichlet_far, 1, (m+1)*(n+1)));
        
        A = sparse(A+dirichlet_far);                    % Values Corresponding to Far-Field Boundary in A Are One on Diagonal
        A(1,1) = 1;                                     % Origin is Always Zero
        
        %% Creating Far-Field Dirichlet Boundary Condition Values
        
        BC = zeros(m+1, n+1);
        
        uppery = ((b+ygrid)/2)-(strike*exp(-r*(0)));    % Updating Boundary Conditions
        upperx = ((xgrid+d)/2)-(strike*exp(-r*(0)));    % Updating Boundary Conditions
        BC(end,:) = upperx;
        BC(:,end) = uppery;
        
        BC = reshape(BC,(m+1)*(n+1), 1);
        
        %% Initial Values for time = T
        
        ICV = max(((X+Y)/2)-strike, 0);
        
        %% 1st Order Time Scheme to Calculate U After First Time Step
        
        U = reshape(ICV, (m+1)*(n+1), 1);
        
        U_minus = U;
        BC_minus = BC;
        
        U = inv((speye(size(A))+(dt*A)))*U_minus+(dt*BC_minus);
        
        U = U-BC;
        
        BC = reshape(BC,(m+1),(n+1));
        upperx = ((xgrid+d)/2)-(strike*exp(-r*(T)));    % Updating Far-Field Boundary Conditions for X
        uppery = ((b+ygrid)/2)-(strike*exp(-r*(T)));    % Updating Far-Field Boundary Conditions for Y
        
        BC(end,:) = upperx;
        BC(:,end) = uppery;
        BC = reshape(BC,(m+1)*(n+1), 1);
        
        U = reshape(U,(m+1),(n+1));
        U(end,:) = 0;
        U(:,end) = 0;
        U = reshape(U,(m+1)*(n+1), 1);
        
        U = U + BC;                                     % Making Sure Far-Field Boundaries Have Correct Value in Case of Rounding Error
        
        %% Calculate Inverse of Matrix Needed for 2nd Order Implicit Time Scheme
        
        A_second_order = inv(A+((1/(2*dt))*speye(size(A))));
        
        %% Time Integration Loop
        % Note: Value is Being Discounted back to the Present from Exersize Date
        
        count = 1;
        len = length(dt : dt : T)-1;
        fprintf("\n %f \n",mult)
        for t = dt : dt : T-dt 
            fprintf("%f ",count)
            fprintf("%f \n",len)
        
            count = count + 1;
        
            top = ((((U-BC))/dt)+((dt/2)*((-2*(U-BC) + (U_minus-BC_minus))/(dt*dt))) + BC); % Updating Implicit Scheme Vector
        
            top = reshape(top,(m+1),(n+1));                 % Making Sure Far-Field Boundaries Have Correct Value in Case of Rounding Error
            top(end,:) = 0;
            top(:,end) = 0;
            top = reshape(top,(m+1)*(n+1), 1);
            top = top + BC;
        
            U_plus = A_second_order*top;                    % 2nd Order Implicit Scheme for Next Time Step
        
            BC_minus = BC;
        
            BC = reshape(BC,(m+1),(n+1));
            upperx = ((xgrid+d)/2)-(strike*exp(-r*(T-t)));  % Updating Far-Field Boundary Conditions for X
            uppery = ((b+ygrid)/2)-(strike*exp(-r*(T-t)));  % Updating Far-Field Boundary Conditions for Y
        
            BC(end,:) = upperx;
            BC(:,end) = uppery;
            BC = reshape(BC,(m+1)*(n+1), 1);
        
            U_plus = reshape(U_plus,(m+1),(n+1));
            U_plus(end,:) = 0;
            U_plus(:,end) = 0;
            U_plus = reshape(U_plus,(m+1)*(n+1), 1);
            
            U_plus = U_plus + BC;
        
            U_minus = U;                                    % Updating U_minus and U for Next Time Step
            U = U_plus;
    
        end
        U_final = max(reshape(U_plus, m+1, n+1), 0);                           % Option Value doesn't go below zero

        x_index = find(xgrid==1.25);
        y_index = find(ygrid==1.25);

        error_list(ii) = U_final(x_index, y_index);

    
    end
end