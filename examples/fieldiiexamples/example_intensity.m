%  Simulate an array using Field II 
%  Calculate the intensity profile along the acoustical axis for
%  the transducer 
% 
%  Version 1.0, 27/6-1997, JAJ 
%  Version 1.1, 27/3-2000, JAJ: Transducer impulse response added
%  Version 1.1, 13/8-2007, JAJ: Change of display during calculation
 
% 
%  Array: 65 elements, width: lambda/2, Height: 5 mm, 
%  Type:  Elevation focused, linear array 

fetal=1;                 %  Whether to use cardiac or fetal intensities

%  Set values for the intensity 

if (fetal==1)

  %  For fetal 
  
  f0=5e6;                  %  Transducer center frequency [Hz] 
  M=2;                     %  Number of cycles in pulse 
  Ispta=0.170*100^2;       %  Fetal intensity: Ispta [w/m^2] 
  %Ispta=0.046*100^2;      %  Fetal intensity In Situ: Ispta [w/m^2] 
  Itype='Fetal';           %  Intensity type used 
else
 
 %  For cardiac 
 
  f0=3e6;                  %  Transducer center frequency [Hz] 
  M=8;                     %  Number of cycles in pulse 
  Ispta=0.730*100^2;       %  Cardiac intensity: Ispta [w/m^2] 
  Itype='Cardiac';         %  Intensity type used 
  end

%  Generate the transducer apertures for send and receive 
 
fs=250e6;                %  Sampling frequency [Hz] 
c=1540;                  %  Speed of sound [m/s] 
lambda=c/f0;             %  Wavelength 
width=lambda/2;          %  Width of element 
element_height=5/1000;   %  Height of element [m] 
kerf=lambda/10;          %  Kerf [m] 
focus=[0 0 60]/1000;     %  Fixed focal point [m] 
elefocus=1;              %  Whether to use elevation focus 
Rfocus=40/1000;          %  Elevation focus [m] 
N_elements=65;           %  Number of physical elements 

  
Z=1.480e6;          %  Characteristic acoustic impedance [kg/(m^2 s)] 
Tprf=1/5e3;         %  Pulse repetition frequency [s] 
Tp=M/f0;            %  Pulse duration [s] 
 
P0=sqrt(Ispta*2*Z*Tprf/Tp);   %  Calculate the peak pressure 
 
%  Set the attenuation to 5*0.5 dB/cm, 0.5 dB/[MHz cm] around 
%  f0 and use this: 
 
set_field ('att',2.5*100); 
set_field ('Freq_att',0.5*100/1e6); 
set_field ('att_f0',f0); 
set_field ('use_att',0);          %  Set this flag to one when including attenuation
 
%  Set the sampling frequency 
 
set_sampling(fs); 
 
%  Make the aperture for the rectangles 
 
if (elefocus == 0) 
  ape = xdc_linear_array (N_elements, width, element_height, kerf, 10, 10, focus); 
else 
  ape = xdc_focused_array (N_elements, width, element_height, kerf, Rfocus, 10, 10, focus); 
  end 
 
%  Set the excitation of the aperture 
 
excitation=sin(2*pi*f0*(0:1/fs:M/f0)); 
excitation=excitation.*hanning(length(excitation))'; 
xdc_excitation (ape, excitation); 
 
%  Set the impulse response of the aperture 
 
impulse=sin(2*pi*f0*(0:1/fs:1/f0)); 
impulse=impulse.*hanning(length(impulse))'; 
xdc_impulse (ape, excitation); 
 
%  Find the scaling factor from the peak value 
 
point=[0 0 0]/1000; 
zvalues=(2:2:100)/1000; 
index=1; 
I=0; 
disp('Finding calibration...') 
for z=zvalues 
  point(3)=z; 
  [y,t] = calc_hp(ape,point); 
  I(index)=sum(y.*y)/(2*Z)/fs/Tprf; 
  index=index+1; 
  end 
I_factor=Ispta/max(I); 
 
%  Set the correct scale factor 
 
scale_factor=sqrt(I_factor); 
excitation=scale_factor*excitation; 
xdc_excitation (ape, excitation); 
 
%  Make the calculation in elevation 
 
disp('Finding pressure and intensity.. ') 
point=[0 0 0]/1000; 
zvalues=(1:1:100)/1000; 
index=1; 
I=0; 
Ppeak=0; 
for z=zvalues 
  if rem(z*1000,10)==0
    disp(['Calculating at distance ',num2str(z*1000),' mm'])
    end
  point(3)=z; 
  [y,t] = calc_hp(ape,point); 
  I(index)=sum(y.*y)/(2*Z)/fs/Tprf; 
  Ppeak(index)=max(y); 
  index=index+1; 
  end 
Pmean=sqrt(I*2*Z*Tprf/Tp); 
 
%  Plot the calculated response 
 
figure(1)
subplot(211) 
plot(zvalues*1000,I*1000/(100^2)) 
xlabel('Axial distance [mm]') 
ylabel('Intensity: Ispta  [mW/cm^2]') 
if (elefocus == 0) 
  title(['Focus at ',num2str(focus(3)*1000),' mm, No elevation focus (',Itype,')']) 
else 
  title(['Focus at ',num2str(focus(3)*1000),' mm, elevation focus at ',num2str(Rfocus*1000),' mm (',Itype,')']) 
end 
subplot(212) 
plot(zvalues*1000,Ppeak/1e6) 
xlabel('Axial distance [mm]') 
ylabel('Peak pressure [MPa]') 
 
%  Do the calculation for a single element 
 
figure(2)
xdc_apodization (ape, 0, [zeros(1,floor(N_elements/2)) 1 zeros(1,floor(N_elements/2))]);  
Ppeak_single=0; 
Isingle=0; 
index=1; 
zvalues=0; 
z=0.001/1000; 
factor=(10/1000/z)^(1/200); 
for index=1:200 
  point(3)=z; 
  zvalues(index)=z; 
  [y,t] = calc_hp(ape,point); 
  z=z*factor; 
  Isingle(index)=sum(y.*y)/(2*Z)/fs/Tprf; 
  Ppeak_single(index)=max(y); 
  index=index+1; 
  end 
Pmean=sqrt(I*2*Z*Tprf/Tp); 
clf 
plot(zvalues*1000,Ppeak_single/1e3) 
axis([-0.1 max(zvalues)*1000 0 1.2*max(Ppeak_single)/1e3]) 
xlabel('Axial distance [mm]') 
ylabel('Peak pressure [kPa]') 
if (elefocus == 0) 
  title(['Single element. Focus at ',num2str(focus(3)*1000),' mm, No elevation focus (',Itype,')']) 
else 
  title(['Single element. Focus at ',num2str(focus(3)*1000),' mm, elevation focus at ',num2str(Rfocus*1000),' mm (',Itype,')']) 
end 
 
%  Release the aperture 
 
xdc_free(ape) 
 
