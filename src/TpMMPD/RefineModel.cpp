#include "StdAfx.h"
#include "../uti_phgrm/MICMAC/cCameraModuleOrientation.h"

/** Developpemet based on 
 CARTOSAT-1 DEM EXTRACTION CAPABILITY STUDY OVER SALON AREA
 R. Gachet & P. Favé
 */

class RefineModel
{
private:
    ElCamera* masterCamera;
    ElCamera* slaveCamera;
    
    // the 6 parameters of affinity (for the slave image)
    // colc = a0 + a1 * col + a2 * lig
    // ligc = b0 + b1 * col + b2 * lig
    double a0;
    double a1;
    double a2;
    double b0;
    double b1;
    double b2;
    
    // Tie Points in the Maser Image (in pixel)
    std::vector<Pt2dr> vPtImgMaster;
    // Tie Points in the Maser Image (in pixel)
    std::vector<Pt2dr> vPtImgSlave;
    // Tie Points altitude (à estimer)
    std::vector<double> vZ;
    
public:
    RefineModel(std::string const &aNameFileGridMaster,
                std::string const &aNameFileGridSlave,
                std::string const &aNamefileTiePoints,
                double Zmoy):masterCamera(NULL),slaveCamera(NULL),a0(0),a1(1),a2(0),b0(0),b1(0),b2(1)
    {
        // Loading the GRID file
        ElAffin2D oriIntImaM2C;
        Pt2di Sz(10000,10000);
        masterCamera = new cCameraModuleOrientation(new OrientationGrille(aNameFileGridMaster),Sz,oriIntImaM2C);
        slaveCamera  = new cCameraModuleOrientation(new OrientationGrille(aNameFileGridSlave),Sz,oriIntImaM2C);
        
        // Loading the Tie Points
        std::ifstream fic(aNamefileTiePoints.c_str());
        while(fic.good())
        {
            Pt2dr P1,P2;
            fic >> P1.x >> P1.y >> P2.x >> P2.y;
            if (fic.good())
            {
                vPtImgMaster.push_back(P1);
                vPtImgSlave.push_back(P2);
                vZ.push_back(Zmoy);
            }
        }
        
        std::cout << "Number of tie points : "<<vPtImgMaster.size()<<std::endl;
    }
    
    
    // compute the difference between the Ground Points for a given TiePoints
    // and a given set of parametes (Z and affinity)
    Pt2dr compute2DGroundDifference(Pt2dr const &ptImgMaster,
                               Pt2dr const &ptImgSlave,
                               double aZ,
                               double aA0,
                               double aA1,
                               double aA2,
                               double aB0,
                               double aB1,
                               double aB2)const
    {
        Pt3dr ptTerMaster = masterCamera->ImEtProf2Terrain(ptImgMaster,aZ);
        Pt2dr ptImgSlaveC(aA0 + aA1 * ptImgSlave.x + aA2 * ptImgSlave.y,
                          aB0 + aB1 * ptImgSlave.x + aB2 * ptImgSlave.y);
        Pt3dr ptTerSlave = slaveCamera->ImEtProf2Terrain(ptImgSlaveC,aZ);
        return Pt2dr(ptTerMaster.x - ptTerSlave.x,ptTerMaster.y - ptTerSlave.y);
    }
    
    
    // compute the observation matrix for one iteration and output it in a text file
    void computeObservationMatrix(std::string const &output)
    {
        
        //cElMatCreuseGen * marix = cElMatCreuseGen::StdNewOne(vPtImgMaster.size(),vPtImgMaster.size(),true);
        
        double dZ = 0.5;
        double dA0 = 0.5;
        double dA1 = 0.1;
        double dA2 = 0.1;
        double dB0 = 0.5;
        double dB1 = 0.1;
        double dB2 = 0.1;
        std::ofstream fic(output.c_str());
        
        
        std::ofstream ficA("A.txt");
        ficA << 2*vPtImgMaster.size() << " " << 6+vPtImgMaster.size()<<std::endl;
        std::ofstream ficX("X.txt");
        ficX << 6+vPtImgMaster.size()<<std::endl;
        
        ficX << a0<<std::endl;
        ficX << a1<<std::endl;
        ficX << a2<<std::endl;
        ficX << b0<<std::endl;
        ficX << b1<<std::endl;
        ficX << b2<<std::endl;
        for(size_t i=0;i<vZ.size();++i)
            ficX<< vZ[i]<<std::endl;
        
        std::ofstream ficB("B.txt");
        ficB << 2*vPtImgMaster.size()<<std::endl;
        
        
        
        for(size_t i=0;i<vPtImgMaster.size();++i)
        {
            Pt2dr const &ptImgMaster = vPtImgMaster[i];
            Pt2dr const &ptImgSlave  = vPtImgSlave[i];
            double const Z = vZ[i];
            
            // ecart constate
            Pt2dr D = compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2,b0,b1,b2);
            
            // estimation des derivees parielles
            Pt2dr vdZ  = Pt2dr(1./dZ,1./dZ)  * (D - compute2DGroundDifference(ptImgMaster,ptImgSlave,Z + dZ,a0,a1,a2,b0,b1,b2));
            Pt2dr vdA0 = Pt2dr(1./dA0,1./dA0) * (D - compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0+dA0,a1,a2,b0,b1,b2));
            Pt2dr vdA1 = Pt2dr(1./dA1,1./dA1) * (D - compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1+dA1,a2,b0,b1,b2));
            Pt2dr vdA2 = Pt2dr(1./dA2,1./dA2) * (D - compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2+dA2,b0,b1,b2));
            Pt2dr vdB0 = Pt2dr(1./dB0,1./dB0) * (D - compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2,b0+dB0,b1,b2));
            Pt2dr vdB1 = Pt2dr(1./dB1,1./dB1) * (D - compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2,b0,b1+dB1,b2));
            Pt2dr vdB2 = Pt2dr(1./dB2,1./dB2) * (D - compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2,b0,b1,b2+dB2));
            
            // output observation for X
            fic <<D.x<<" "<<vdA0.x<<" "<<vdA1.x <<" "<<vdA2.x<<" "<<vdB0.x<<" "<<vdB1.x<<" "<<vdB2.x<<" ";
            for(size_t j=0;j<vPtImgSlave.size();++j)
            {
                if (i!=j)
                    fic << "0.0 ";
                else
                    fic << vdZ.x<<" ";
            }
            fic << std::endl;
            
            // output observation for Y
            fic <<D.y<<" "<<vdA0.y<<" "<<vdA1.y <<" "<<vdA2.y<<" "<<vdB0.y<<" "<<vdB1.y<<" "<<vdB2.y<<" ";
            for(size_t j=0;j<vPtImgSlave.size();++j)
            {
                if (i!=j)
                    fic << "0.0 ";
                else
                    fic << vdZ.y<<" ";
            }
            fic << std::endl;
            
            ficA << 2*i << " "<<0<<" "<<vdA0.x<<std::endl;
            ficA << 2*i << " "<<1<<" "<<vdA1.x<<std::endl;
            ficA << 2*i << " "<<2<<" "<<vdA2.x<<std::endl;
            ficA << 2*i << " "<<3<<" "<<vdB0.x<<std::endl;
            ficA << 2*i << " "<<4<<" "<<vdB1.x<<std::endl;
            ficA << 2*i << " "<<5<<" "<<vdB2.x<<std::endl;
            ficA << 2*i << " "<<6+i<<" "<<vdZ.x<<std::endl;
            
            ficB << D.x<<std::endl;
            
            ficA << 2*i+1 << " "<<0<<" "<<vdA0.y<<std::endl;
            ficA << 2*i+1 << " "<<1<<" "<<vdA1.y<<std::endl;
            ficA << 2*i+1 << " "<<2<<" "<<vdA2.y<<std::endl;
            ficA << 2*i+1 << " "<<3<<" "<<vdB0.y<<std::endl;
            ficA << 2*i+1 << " "<<4<<" "<<vdB1.y<<std::endl;
            ficA << 2*i+1 << " "<<5<<" "<<vdB2.y<<std::endl;
            ficA << 2*i+1 << " "<<6+i<<" "<<vdZ.y<<std::endl;
            
            ficB << D.y<<std::endl;
            
        }
    }
    
    
    ~RefineModel()
    {
        if (masterCamera)
            delete masterCamera;
        if (slaveCamera)
            delete slaveCamera;
    }
};


int RefineModel_main(int argc, char **argv) {
    std::string aNameFileGridMaster;// fichier GRID image maitre
    std::string aNameFileGridSlave;// fichier GRID image secondaire
    std::string aNameFileTiePoints;// fichie de points de liaison
    double aZMoy;//the average altitude of the TiePoints
    
    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aNameFileGridMaster,"master image GRID")
                   << EAMC(aNameFileGridSlave,"slave image GRID")
                   << EAMC(aNameFileTiePoints,"Tie Points")
                   << EAMC(aZMoy,"average altitude of the TiePoints")
     ,
        LArgMain()
     );

    RefineModel model(aNameFileGridMaster,aNameFileGridSlave,aNameFileTiePoints,aZMoy);
    model.computeObservationMatrix("matrix.txt");
    
    
       return EXIT_SUCCESS;
}
