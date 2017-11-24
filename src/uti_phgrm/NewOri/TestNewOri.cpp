/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

#include "NewOri.h"


/*
int TestNewOriImage_main(int argc,char ** argv)
{
   std::string aNameOri,aNameI1,aNameI2;
   ElInitArgMain
   (
        argc,argv,
        LArgMain() <<  EAMC(aNameI1,"Name First Image")
                   <<  EAMC(aNameI2,"Name Second Image"),
        LArgMain() << EAM(aNameOri,"Ori",true,"Orientation ")
   );


    cNewO_NameManager aNM("./",aNameOri,"dat");

    CamStenope * aC1 = aNM.CamOfName(aNameI1);
    CamStenope * aC2 = aNM.CamOfName(aNameI2);

    ElPackHomologue aLH = aNM.PackOfName(aNameI1,aNameI2);

    std::cout << "FFF " << aC1->Focale() << " " << aC2->Focale() << " NBh : " << aLH.size() << "\n";

    return EXIT_SUCCESS;
}
*/

///Export the graph to G2O format for testing in ceres
class cAppliNOExport;
class Constraint;
class Pose3d;
class RotMat;

//namespace ceresTestER
//{
    class RotMat 
    {
        public :
            RotMat(const Pt3dr &aI,const Pt3dr &aJ,const Pt3dr &aK) : 
                mI(aI), 
                mJ(aJ),
                mK(aK) {};
            
            Pt3dr & I(){return mI;}
            Pt3dr & J(){return mJ;}
            Pt3dr & K(){return mK;}

        private :
            Pt3dr   mI;
            Pt3dr   mJ;
            Pt3dr   mK;
    };

    class Pose3d
    {
        public :
            Pose3d(const Pt3dr &aP,
                   const Pt3dr &aI,const Pt3dr &aJ,const Pt3dr &aK,
                   const int aId) : 
                mP(aP),
                mQ(aI,aJ,aK),
                mId(aId) {};

            int    & Id(){return mId;};
            Pt3dr  & P(){return mP;}
            RotMat & R(){return mQ;}

            static std::string name() {return "VERTEX_SE3:QUAT";};
        
        private : 
            Pt3dr              mP;
            RotMat             mQ;
            int                mId;


    };
 
    class Constraint
    {
        public :
            Constraint(const int & aI0,const int & aI1,
                       const Pose3d aRel,
                       const Pt3dr  aPdsT,
                       const Pt3dr  aPdsR) : 
                mI0(aI0),
                mI1(aI1),
                mRel(aRel),
                mPdsR(aPdsR),
                mPdsT(aPdsT) {};
                
            int    &  I0(){return mI0;};
            int    &  I1(){return mI1;};
            Pose3d &  Pose(){return mRel;};
            Pt3dr  &  PdsR(){return mPdsR;}
            Pt3dr  &  PdsT(){return mPdsT;}

            static std::string name() {return "EDGE_SE3:QUAT";}
                
        private :
            int      mI0,mI1; 
            Pose3d   mRel;
            Pt3dr    mPdsR;
            Pt3dr    mPdsT;
            
    };
    
    class cAppliNOExport : public cCommonMartiniAppli
    {
        public : 
            cAppliNOExport(int argc,char ** argv);

        private :
            bool NOSave(const std::map<std::string,Pose3d *> aMP,
                        const std::vector<Constraint*> aCVec,
                        const std::string & aName );
            void NOSaveConstraint(std::fstream* aFile, Constraint* aC);
            void NOSaveNoed(std::fstream* aFile,Pose3d* aMP);
    };
    

    ///pose X Y Z rot_I_x rot_I_y rot_I_z rot_J_x rot_J_y rot_J_z rot_K_x rot_K_y rot_K_z
    void cAppliNOExport::NOSaveNoed(std::fstream* aFile,Pose3d* aMP)
    {
        *aFile << aMP->name().c_str() << " " << aMP->Id() << " " << aMP->P().x << " " << aMP->P().y << " " << aMP->P().z << 
           " " << aMP->R().I().x << " " << aMP->R().I().y << " " << aMP->R().I().z << 
           " " << aMP->R().J().x << " " << aMP->R().J().y << " " << aMP->R().J().z << 
           " " << aMP->R().K().x << " " << aMP->R().K().y << " " << aMP->R().K().z << "\n";
    };

    ///ID_a ID_b x_ab y_ab z_ab q_x_ab q_y_ab q_z_ab q_w_ab I_11 I_12 I_13 ... I_16 I_22 I_23 ... I_26 ... I_66 
    void cAppliNOExport::NOSaveConstraint(std::fstream* aFile, Constraint* aC)
    {
        *aFile << aC->name().c_str() << " " << aC->I0() << " " << aC->I1() << " " 
               << aC->Pose().P().x << " " << aC->Pose().P().y << " " << aC->Pose().P().z << " "
               << aC->Pose().R().I().x << " " << aC->Pose().R().I().y << " " << aC->Pose().R().I().z << " "
               << aC->Pose().R().J().x << " " << aC->Pose().R().J().y << " " << aC->Pose().R().J().z << " "
               << aC->Pose().R().K().x << " " << aC->Pose().R().K().y << " " << aC->Pose().R().K().z << " "
               << aC->PdsR().x << " 0 0 0 0 0 " 
               << "0 " << aC->PdsR().y << " 0 0 0 0 "
               << "0 0 " << aC->PdsR().z << " 0 0 0 "
               << "0 0 0 " << aC->PdsT().x << " 0 0 " 
               << "0 0 0 0 " << aC->PdsT().y << " 0 "
               << "0 0 0 0 0 " << aC->PdsT().z << "\n";

                
    };

    bool cAppliNOExport::NOSave(const std::map<std::string,Pose3d *> aMP,
              const std::vector<Constraint*> aCVec,
              const std::string & aName )
    {
       std::fstream aOut;
       aOut.open(aName.c_str(), std::istream::out); 

       if (!aOut) 
       {
            ELISE_ASSERT
            (
                    false,
                    "NewOriImage2G2O_main save; can't open file"
            );

            return false;
       }

       for(auto aK : aMP)
       {
           NOSaveNoed(&aOut,aK.second);
       }

       for(auto aK : aCVec)
       {
           NOSaveConstraint(&aOut,aK);
       }

       return true;

    };




    cAppliNOExport::cAppliNOExport(int argc,char ** argv) :
        cCommonMartiniAppli ()
    {

        std::string aPat,aDir;
        std::string aOri="Martini";
        std::string aName="triplets_g2o.txt";
 
        cInterfChantierNameManipulateur * aICNM;
        std::list<std::string> aLFile;

        ElInitArgMain
        (
            argc,argv,
            LArgMain() << EAMC(aPat,"Pattern of images", eSAM_IsExistFile),
            LArgMain() << EAM(aOri,"Ori",true,"Initial absolute ori; Def=[Ori-Martini]")
                       << EAM(aName,"Out",true,"Output file name")
        );
 
        #if (ELISE_windows)
            replace( aPat.begin(), aPat.end(), '\\', '/' );
        #endif
 
        SplitDirAndFile(aDir,aPat,aPat);
        StdCorrecNameOrient(aOri,aDir);
 
        /// get map of initial orientations
        aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
        aOri= aICNM->StdKeyOrient(aOri);
        aLFile =  aICNM->StdGetListOfFile(aPat,1);

        std::map<std::string,Pose3d *> aMP;

        int aNCP=0;
        for( auto aL : aLFile )
        {

            std::string aNF = aICNM->Dir() + aICNM->Assoc1To1(aOri,aL,true);
            Pt3dr aC = StdGetObjFromFile<Pt3dr>
                    (
                        aNF,
                        StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                        "Centre",
                        "Pt3dr"
                    );        
    
            cOrientationConique * aCO = OptionalGetObjFromFile_WithLC<cOrientationConique>
                                     (
                                           0,0,
                                           aNF,
                                           StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                           "OrientationConique",
                                           "OrientationConique"
                                     );
            cRotationVect aRV   = aCO->Externe().ParamRotation();
            
            std::cout << "+P(" << aL << ")=" << aNCP << "\n";

            aMP[aL] = new Pose3d(aC,
                             aRV.CodageMatr().Val().L1(),
                             aRV.CodageMatr().Val().L2(),
                             aRV.CodageMatr().Val().L3(),
                             aNCP++);
        
        }        
        std::cout << "No de noeds=" << aNCP << "\n";
 

        ///triplets dir manager
        cNewO_NameManager *  aNM = NM(aDir);
        std::string aNameLTriplets = aNM->NameTopoTriplet(true);
        cXml_TopoTriplet  aLT = StdGetFromSI(aNameLTriplets,Xml_TopoTriplet);
        

        ///get vector of constraints
        int aNCC=0;
        std::vector<Constraint*> aCVec;
        
        for (auto a3 : aLT.Triplets())
        {
            if (DicBoolFind(aMP,a3.Name1()) && DicBoolFind(aMP,a3.Name2()) && DicBoolFind(aMP,a3.Name3()))
            {
                std::string  aName3R = aNM->NameOriOptimTriplet(true,a3.Name1(),a3.Name2(),a3.Name3());
                cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);
                
                //find id inside aMP
                Pose3d *aPA1 = aMP[a3.Name1()];
                Pose3d *aPA2 = aMP[a3.Name2()];
                Pose3d *aPA3 = aMP[a3.Name3()];
         
        
 
                ///1-2
                ElRotation3D aP12 = Xml2El(aXml3Ori.Ori2On1());      
                
                Pt3dr aI,aJ,aK;
                aP12.Mat().GetCol(0,aI);
                aP12.Mat().GetCol(1,aJ);
                aP12.Mat().GetCol(2,aK);
         
                
                aCVec.push_back(new Constraint( aPA1->Id(),aPA2->Id(), 
                                                Pose3d(aP12.tr(),aI,aJ,aK,aNCC++),
                                                Pt3dr(1,1,1),
                                                Pt3dr(1,1,1)
                                                ));
           
                std::cout << "C=(" << aPA1->Id() << "," << aPA2->Id() << ")="  << a3.Name1() << "-" << a3.Name2() << "\n"; 
                ///1-3
                ElRotation3D aP13 = Xml2El(aXml3Ori.Ori3On1());
             
                aP13.Mat().GetCol(0,aI);
                aP13.Mat().GetCol(1,aJ);
                aP13.Mat().GetCol(2,aK);
             
                aCVec.push_back(new Constraint( aPA1->Id(),aPA3->Id(), 
                                                Pose3d(aP13.tr(),aI,aJ,aK,aNCC++),
                                                Pt3dr(1,1,1),
                                                Pt3dr(1,1,1)));
                
                std::cout << "C=(" << aPA1->Id() << "," << aPA3->Id() << ")=" << a3.Name1() << "-" << a3.Name3() << "\n"; 
            }
        }
        std::cout << "No de contraints=" << aNCC << "\n";



       
        NOSave(aMP,aCVec,aName);
                
    }

    int CPP_NewOriImage2G2O_main(int argc,char ** argv)
    {
        cAppliNOExport aAppli(argc,argv);
        return EXIT_SUCCESS;

    }
//}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
