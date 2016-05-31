% ExtractAsterXML v0.2
%
% This function reads an ASTER HDF file, extracts and destrip an image from
% a choosen band to tif and extracts the associated geolocalization meta data
% to an xml compliant with IGN's MICMAC's format.
% 
% Syntax:
% ExtractAsterXML(imName,band)
% imName is the Aster hdf file
% band is the name of the band to extract (1,2,3N,3B)
%
% Luc Girod 2015.08.28
% luc.girod@geo.uio.no
% University of Oslo
%
%

function [] = ExtractAsterXML(imName,band)

%% Extract and destrip images

ImageData = double(hdfread(imName, ['/VNIR/VNIR_Band' band '/Data Fields/ImageData'], 'Index', {[],[],[]}));
CorFact = double(hdfread(imName, ['/VNIR/VNIR_Band' band '/Data Fields/RadiometricCorrTable'], 'Index', {[],[],[]}));

% Applying radiometric correction table

for i=1:size(CorFact,1)
    imCor(:,i)=CorFact(i,2)*ImageData(:,i)/CorFact(i,3)+CorFact(i,1);
end
imwrite(uint8(imCor),[imName '_' band '.tif'])


%% Create xml file
clear AsterMeta aXML LatticePoints
aXML = com.mathworks.xml.XMLUtils.createDocument('AsterMeta');
AsterMeta = aXML.getDocumentElement;
AsterMeta.setAttribute('version','0.2');


%% Lattice points
LatticePoint = hdfread(imName, ['/VNIR/VNIR_Band' band '/Data Fields/LatticePoint'], 'Index', {[],[],[]});

LatticePoints = aXML.createElement('LatticePoints');

        NbLattice = aXML.createElement('NbLattice'); 
        NbLattice.appendChild(aXML.createTextNode(num2str(size(LatticePoint,1)*size(LatticePoint,2))));
        LatticePoints.appendChild(NbLattice);
k=0;       
for i = 1:size(LatticePoint,1)
    for j=1:size(LatticePoint,2)
        k=k+1;
        aLatticePt = aXML.createElement(['LatticePoint_' num2str(k)]); 
        aLatticePt.appendChild(aXML.createTextNode([num2str(LatticePoint(i,j,1),100) ' ' num2str(LatticePoint(i,j,2),100)]));
        LatticePoints.appendChild(aLatticePt);
    end
end

AsterMeta.appendChild(LatticePoints);


%% Sattelite Positions
SatellitePosition = hdfread(imName, ['/VNIR/VNIR_Band' band '/Data Fields/SatellitePosition'], 'Index', {[],[],[]});

SatellitePositions = aXML.createElement('SatellitePositions');

        NbSatPos = aXML.createElement('NbSatPos'); 
        NbSatPos.appendChild(aXML.createTextNode(num2str(size(SatellitePosition,1))));
        SatellitePositions.appendChild(NbSatPos);
        
for i = 1:size(SatellitePosition,1)
    aSatPos = aXML.createElement(['SatPos_' num2str(i)]); 
    aSatPos.appendChild(aXML.createTextNode([num2str(SatellitePosition(i,1),100) ' ' num2str(SatellitePosition(i,2),100) ' ' num2str(SatellitePosition(i,3),100)]));
    SatellitePositions.appendChild(aSatPos);
end

AsterMeta.appendChild(SatellitePositions);


%% Lattitude Longitude -> ECEF
Longitude = hdfread(imName, ['/VNIR/VNIR_Band' band '/Geolocation Fields/Longitude'], 'Index', {[],[],[]});
Latitude = hdfread(imName, ['/VNIR/VNIR_Band' band '/Geolocation Fields/Latitude'], 'Index', {[],[],[]});

%Convert points to ECEF (geocentric euclidian)
a = 6378137;
b = (1 - 1 / 298.257223563)*a;
ECEF=[];
for i = 1:size(Latitude,1)
    for j=1:size(Latitude,2)
        	aSinLat = sin(Latitude(i,j)*pi / 180);
			aCosLat = cos(Latitude(i,j)*pi / 180);
			aSinLon = sin(Longitude(i,j)*pi / 180);
			aCosLon = cos(Longitude(i,j)*pi / 180);
			r = sqrt(a*a*b*b / (a*a*aSinLat*aSinLat + b*b*aCosLat*aCosLat));
			x = r*aCosLat*aCosLon;
			y = r*aCosLat*aSinLon;
			z = r*aSinLat;
            ECEF=[ECEF;x y z];
    end
end



ECEFs = aXML.createElement('ECEFs');

        NbECEF = aXML.createElement('NbECEF'); 
        NbECEF.appendChild(aXML.createTextNode(num2str(size(ECEF,1))));
        ECEFs.appendChild(NbECEF);
      
for i = 1:size(ECEF,1)
    aECEF = aXML.createElement(['ECEF_' num2str(i)]); 
    aECEF.appendChild(aXML.createTextNode([num2str(ECEF(i,1),100) ' ' num2str(ECEF(i,2),100) ' ' num2str(ECEF(i,3),100)]));
    ECEFs.appendChild(aECEF);
end

AsterMeta.appendChild(ECEFs);

xmlwrite([imName '_' band '.xml'],aXML);
