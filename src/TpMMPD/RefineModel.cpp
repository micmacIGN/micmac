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
	double zMoy;
    
public:
    RefineModel(std::string const &aNameFileGridMaster,
                std::string const &aNameFileGridSlave,
                std::string const &aNamefileTiePoints,
                double Zmoy):masterCamera(NULL),slaveCamera(NULL),a0(0),a1(1),a2(0),b0(0),b1(0),b2(1),zMoy(Zmoy)
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
			// double Z; // MPD : unused
           // fic >> P1.x >> P1.y >> P2.x >> P2.y >> Z;
			 fic >> P1.x >> P1.y >> P2.x >> P2.y ;//>> Z;
            if (fic.good())
            {
                vPtImgMaster.push_back(P1);
                vPtImgSlave.push_back(P2);
                vZ.push_back(Zmoy);
				//vZ.push_back(Z);
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

	
	 double sumRes()
	 {
		 bool verbose = true;

		 double sumRes = 0.;
		 for(size_t i=0;i<vPtImgMaster.size();++i)
         {
			Pt2dr const &ptImgMaster = vPtImgMaster[i];
            Pt2dr const &ptImgSlave  = vPtImgSlave[i];
            double const Z = vZ[i];
            
            // ecart constate
            Pt2dr D = compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2,b0,b1,b2);

			if (verbose) std::cout << D << std::endl;

			sumRes += D.x*D.x + D.y*D.y;
		 }
		 return sumRes;
	 }
    
    
    // compute the observation matrix for one iteration and output it in a text file
    void computeObservationMatrix(std::string const &output)
    {
		size_t numUnk = 6+vPtImgMaster.size();  //Nombre d'inconnues 
		size_t numObs = 2*vPtImgMaster.size() + numUnk;  //Nombre d'observations (dont stabilisation des inconnues)
		
		double curRMS = std::sqrt(sumRes()/numObs);
		
		std::cout << "RMS_ini = " << curRMS << std::endl;

		
        
        //cElMatCreuseGen * marix = cElMatCreuseGen::StdNewOne(vPtImgMaster.size(),vPtImgMaster.size(),true);
        
        double dZ = 0.5;
        double dA0 = 0.5;
        double dA1 = 0.1;
        double dA2 = 0.1;
        double dB0 = 0.5;
        double dB1 = 0.1;
        double dB2 = 0.1;


		//Ponderation
		double sigmaDelta = 1./std::pow(1.,2); //m
		bool   weightByRes = false;
		//Ponderation stabilisation
		double sigmaTransX = 1./std::pow(1.,2); //pix
		double sigmaTransY = 1./std::pow(1.,2);  //pix
		double sigmaMat = 1./std::pow(0.001,2); //sans unite
		double sigmaZ = 1./std::pow(100.,2);  //m
		
		//bool   filter      = false; //todo


		std::ofstream ficP("P.txt");
		ficP << numObs <<std::endl;
       
        std::ofstream ficA("A.txt");
        ficA << numObs  << " " << numUnk <<std::endl;
        
        std::ofstream ficB("B.txt");
        ficB << numObs <<std::endl;
        
        
        
        for(size_t i=0;i<vPtImgMaster.size();++i)
        {
            Pt2dr const &ptImgMaster = vPtImgMaster[i];
            Pt2dr const &ptImgSlave  = vPtImgSlave[i];
            double const Z = vZ[i];
            
            // ecart constate
            Pt2dr D = compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2,b0,b1,b2);

			//todo : strategie d'elimination d'observations / ou ponderation
            
            // estimation des derivees partielles
            Pt2dr vdZ  = -Pt2dr(1./dZ,1./dZ)  * (D - compute2DGroundDifference(ptImgMaster,ptImgSlave,Z + dZ,a0,a1,a2,b0,b1,b2));
            Pt2dr vdA0 = -Pt2dr(1./dA0,1./dA0) * (D - compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0+dA0,a1,a2,b0,b1,b2));
            Pt2dr vdA1 = -Pt2dr(1./dA1,1./dA1) * (D - compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1+dA1,a2,b0,b1,b2));
            Pt2dr vdA2 = -Pt2dr(1./dA2,1./dA2) * (D - compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2+dA2,b0,b1,b2));
            Pt2dr vdB0 = -Pt2dr(1./dB0,1./dB0) * (D - compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2,b0+dB0,b1,b2));
            Pt2dr vdB1 = -Pt2dr(1./dB1,1./dB1) * (D - compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2,b0,b1+dB1,b2));
            Pt2dr vdB2 = -Pt2dr(1./dB2,1./dB2) * (D - compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2,b0,b1,b2+dB2));
            
  
            ficA << 2*i << " "<<0<<" "<<vdA0.x<<std::endl;
            ficA << 2*i << " "<<1<<" "<<vdA1.x<<std::endl;
            ficA << 2*i << " "<<2<<" "<<vdA2.x<<std::endl;
            ficA << 2*i << " "<<3<<" "<<vdB0.x<<std::endl;
            ficA << 2*i << " "<<4<<" "<<vdB1.x<<std::endl;
            ficA << 2*i << " "<<5<<" "<<vdB2.x<<std::endl;			
            ficA << 2*i << " "<<6+i<<" "<<vdZ.x<<std::endl;
            
            ficB << -D.x<<std::endl;

			if (!weightByRes)
				ficP << sigmaDelta << std::endl;
			else
			{
				double sigmaD = 1./ (std::sqrt(D.x*D.x+D.y*D.y)); 
				ficP << sigmaD << std::endl;
			}
            
            ficA << 2*i+1 << " "<<0<<" "<<vdA0.y<<std::endl;
            ficA << 2*i+1 << " "<<1<<" "<<vdA1.y<<std::endl;
            ficA << 2*i+1 << " "<<2<<" "<<vdA2.y<<std::endl;
            ficA << 2*i+1 << " "<<3<<" "<<vdB0.y<<std::endl;
            ficA << 2*i+1 << " "<<4<<" "<<vdB1.y<<std::endl;
            ficA << 2*i+1 << " "<<5<<" "<<vdB2.y<<std::endl;
            ficA << 2*i+1 << " "<<6+i<<" "<<vdZ.y<<std::endl;
            
            ficB << -D.y<<std::endl;
			
			if (!weightByRes)
				ficP << sigmaDelta << std::endl;
			else
			{
				double sigmaD = 1./ (std::sqrt(D.x*D.x+D.y*D.y)); 
				ficP << sigmaD << std::endl;
			}
            
        }
				

		//equations de stabilisation
		size_t numObsTmp = 2*vPtImgMaster.size();
		ficA << numObsTmp++ << " " << 0 << "  " << 1. << std::endl;
		ficA << numObsTmp++ << " " << 1 << "  " << 1. << std::endl;
		ficA << numObsTmp++ << " " << 2 << "  " << 1. << std::endl;
		ficA << numObsTmp++ << " " << 3 << "  " << 1. << std::endl;
		ficA << numObsTmp++ << " " << 4 << "  " << 1. << std::endl;
		ficA << numObsTmp++ << " " << 5 << "  " << 1. << std::endl;
		for(size_t k = 0; k < vPtImgMaster.size();k++)
			ficA << numObsTmp++ << " " << 6+k << "  " << 1. << std::endl;

		ficB << 0.-a0 << std::endl;		ficP << sigmaTransX << std::endl;
		ficB << 1.-a1 << std::endl;		ficP << sigmaMat << std::endl;
		ficB << 0.-a2 << std::endl;		ficP << sigmaMat << std::endl;
		ficB << 0.-b0 << std::endl;		ficP << sigmaTransY << std::endl;
		ficB << 0.-b1 << std::endl;		ficP << sigmaMat << std::endl;
		ficB << 1.-b2 << std::endl;		ficP << sigmaMat << std::endl;
		for(size_t k = 0; k < vPtImgMaster.size();k++)
		{
			//ficB << 0.-vZ[k] << std::endl;  //todo : zIni != 0
			ficB << zMoy-vZ[k] << std::endl;  //todo : zIni != 0
			ficP << sigmaZ << std::endl;
		}

		ficA.close();
		ficB.close();
		ficP.close();
		

		//Resolution du systeme
		std::string cmd = "tool-IgnSocle_math_solveSparseLS --A A.txt --B B.txt --X X.txt --P P.txt";
		//system(cmd.c_str());  // MPD : attention : ignoring return value of ‘int system(const char*)’
		System(cmd);

		//Recuperation de la solution

		 std::ifstream ficSol("X.txt");
		 size_t toto;
	
		 double sumD = 0.;
		 ficSol >> toto; 
		 ficSol >> dA0; a0 += dA0; sumD += dA0*dA0;
		 ficSol >> dA1; a1 += dA1; sumD += dA1*dA1;
		 ficSol >> dA2; a2 += dA2; sumD += dA2*dA2;
		 ficSol >> dB0; b0 += dB0; sumD += dB0*dB0;
		 ficSol >> dB1; b1 += dB1; sumD += dB1*dB1;
		 ficSol >> dB2; b2 += dB2; sumD += dB2*dB2;		 
		  for(size_t i=0;i<vPtImgMaster.size();++i)
		  {
			  ficSol >> dZ; vZ[i] += dZ; sumD += dZ*dZ;
		  }
		std::cout << "sumDiff = " << sumD << std::endl;


		
		//std::cout << "sumRes_after = " << sumRes() << std::endl;
		curRMS = std::sqrt(sumRes()/numObs);
		
		std::cout << "RMS_after = " << curRMS << std::endl;

		


		std::cout << " Sol  = " << a0 << " " << a1 << " " << a2 << " " << b0 << " " << b1 << " " << b2 << std::endl;
		std::cout << " SolZ  = " ;
		for(size_t i=0;i<vPtImgMaster.size();++i)
			std::cout << vZ[i] << " ";
		std::cout << std::endl;
        
    

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

	for(size_t iter = 0; iter < 15; iter++)
		model.computeObservationMatrix("matrix.txt");
    
    
       return EXIT_SUCCESS;
}
