clear; clc; close all;

%% ============================================================
% Quintet (S = 2) powder ODMR simulation
% High-field basis initial populations and optical readout
% Including powder orientation averaging
% =============================================================

%% Experimental parameters
mwFreq = 9.58e3;          % Microwave frequency [MHz]
BRange = [266 421];       % Magnetic field range [mT]
nB = 1000;
Blist = linspace(BRange(1), BRange(2), nB);

g = 2.0023;
muB_h = 13.996245;        % muB/h [MHz/mT]

%% Quintet spin parameters
S = 2;
D = 320;                  % ZFS parameter D [MHz]
E = -20;                  % ZFS parameter E [MHz]

%% Spectral parameters
linewidth = 30;           % Lorentzian HWHM [MHz]
nTheta = 30;
nPhi = 60;

%% Initial populations in high-field basis
% Order: mS = +2, +1, 0, -1, -2

pHigh = [0.3 0.0 0.4 0.0 0.3];

pHigh = pHigh / sum(pHigh);

%% Optical emission intensity in high-field basis
% Order: mS = +2, +1, 0, -1, -2

plHigh = [0.3 0.0 0.4 0.0 0.3];

%% Spin operators
m = S:-1:-S;
dim = length(m);

Sz = diag(m);

Sp = zeros(dim);

for k = 1:dim-1
    Sp(k,k+1) = sqrt(S*(S+1)-m(k+1)*m(k));
end

Sm = Sp';

Sx = (Sp + Sm)/2;
Sy = (Sp - Sm)/(2i);

Id = eye(dim);

%% Zero-field splitting Hamiltonian

H_ZFS = D*(Sz^2 - S*(S+1)/3*Id) ...
      + E*(Sx^2 - Sy^2);

%% Powder orientation grid

thetaList = linspace(0,pi,nTheta);
phiList = linspace(0,2*pi,nPhi+1);
phiList(end) = [];

%% Initialize spectrum

spectrum = zeros(1,nB);

%% ============================================================
% Orientation averaging
% =============================================================

for it = 1:nTheta

    theta = thetaList(it);

    for ip = 1:nPhi

        phi = phiList(ip);

        %% Magnetic field direction

        nx = sin(theta)*cos(phi);
        ny = sin(theta)*sin(phi);
        nz = cos(theta);

        %% Powder averaging weight

        weight = sin(theta);

        %% Rotation to the high-field quantization axis
        % The high-field quantization axis is parallel to B.
        %
        % U = exp(-i*phi*Sz)*exp(-i*theta*Sy)
        %
        % This transforms the laboratory z-axis into the
        % magnetic-field direction.

        U = expm(-1i*phi*Sz) * expm(-1i*theta*Sy);

        %% High-field basis for this orientation

        VHigh = U;

        %% Initial density matrix in high-field basis

        rho0 = VHigh * diag(pHigh) * VHigh';

        %% Optical readout operator in high-field basis

        PL_operator = VHigh * diag(plHigh) * VHigh';

        %% Microwave polarization directions perpendicular to B

        e1 = [cos(theta)*cos(phi), ...
              cos(theta)*sin(phi), -sin(theta)];

        e2 = [-sin(phi), cos(phi), 0];

        Smw1 = e1(1)*Sx + e1(2)*Sy + e1(3)*Sz;

        Smw2 = e2(1)*Sx + e2(2)*Sy + e2(3)*Sz;

        %% Spin operator along magnetic field

        SB = nx*Sx + ny*Sy + nz*Sz;

        %% Eigenstates of spin operator along B

        [UB,MB] = eig(SB);

        [mB_values,idxB] = sort(diag(MB),'descend');

        UB = UB(:,idxB);

        %% Magnetic field sweep

        for ib = 1:nB

            B = Blist(ib);

            %% Hamiltonian [MHz]

            H = H_ZFS + g*muB_h*B*SB;

            %% Diagonalize Hamiltonian

            [V,Eval] = eig(H);

            [energy,idx] = sort(diag(Eval));

            V = V(:,idx);

            %% Determine dominant mS character

            mB_character = abs(UB' * V).^2;

            [~,maxIdx] = max(mB_character,[],1);

            mB_state = round(mB_values(maxIdx));

            %% Populations in field eigenbasis

            pB = real(diag(V' * rho0 * V));

            %% Optical emission intensity in field eigenbasis

            plB = real(diag(V' * PL_operator * V));

            %% Calculate all transitions

            for i = 1:dim-1

                for j = i+1:dim

                    % Selection rule: Delta mS = +/- 1

                    if abs(mB_state(j)-mB_state(i)) ~= 1
                        continue;
                    end

                    % Transition frequency

                    freq = abs(energy(j)-energy(i));

                    % Microwave transition probability

                    M1 = abs(V(:,i)'*Smw1*V(:,j))^2;

                    M2 = abs(V(:,i)'*Smw2*V(:,j))^2;

                    W = (M1+M2)/2;

                    % Population transfer contribution

                    population_factor = pB(i)-pB(j);

                    % Optical readout contribution

                    optical_factor = plB(j)-plB(i);

                    % Lorentzian resonance

                    detuning = freq-mwFreq;

                    L = linewidth^2 / ...
                        (detuning^2+linewidth^2);

                    % Signed ODMR contribution

                    spectrum(ib) = spectrum(ib) ...
                        + weight*W ...
                        * population_factor ...
                        * optical_factor*L;

                end
            end
        end
    end
end

%% Normalize spectrum

spectrum = spectrum / max(abs(spectrum));

%% Plot

figure('Color','w');

plot(Blist,spectrum,'k-','LineWidth',1.5);

xlabel('Magnetic field (mT)');
ylabel('Normalized ODMR signal');

title('Quintet (S = 2) powder ODMR');

xlim(BRange);

set(gca,'FontSize',13,'LineWidth',1.2);

box on;
yline(0,'--');

%% Export simulated spectrum as CSV

outputData = [Blist(:), spectrum(:)];

filename = 'X-band_Quintet_ODMR_highfield_basis_all-transitions.csv';

writematrix(outputData,filename);

disp(['Spectrum saved as: ',filename]);