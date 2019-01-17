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


/// Classe qui contient l'orientation relative de chaque image dans un triplet et son sommet GPS
class cGpsLoc_Triplet;

/// Classe qui contient le nom d'une image et son repere GPS
class cGpsLoc_Som;     
typedef cGpsLoc_Som *  tGLS_Ptr;

///  Classe pour definir un repere image avec juste 1 Point et 3 vecteur  
class cGpsLoc_Rep;

/// Classe de l'application
class cAppliGpsLoc;


class cSolBasculeRig  ;
cSolBasculeRig StdSolFromPts
                (
                      const std::vector<Pt3dr> & aV1,
                      const std::vector<Pt3dr> & aV2,
                      const std::vector<double> * aVPds=0, // si 0 ts les pds valent 1
                      int   aNbRansac             = 200,
                      int   aNbL2                 = 5
                );

/** 
   cXml_Rotation
ElRotation3D Xml2El(const cXml_Rotation & aXml)
cXml_Rotation El2Xml(const ElRotation3D & aRot);

*/
 
/* ========================================= */
/*                                           */
/*         cGpsLoc_Rep                       */
/*                                           */
/* ========================================= */



class cGpsLoc_Rep
{
     public :
        cGpsLoc_Rep(); // Pour rotation identitie, et centre optique à (0,0,0)
        cGpsLoc_Rep(const ElRotation3D & aPose);
        ElMatrix<double>  MatRot() const;
        const Pt3dr & Ori() const {return mOri;}
     private :
         Pt3dr mOri;
         Pt3dr mI;
         Pt3dr mJ;
         Pt3dr mK;
     
};

cGpsLoc_Rep::cGpsLoc_Rep () :
    mOri (0,0,0),
    mI   (1,0,0),
    mJ   (0,1,0),
    mK   (0,0,1)
{
}

cGpsLoc_Rep::cGpsLoc_Rep (const ElRotation3D & aPose) :
    mOri (aPose.ImAff(Pt3dr(0,0,0))),
    mI   (aPose.ImVect(Pt3dr(1,0,0))),
    mJ   (aPose.ImVect(Pt3dr(0,1,0))),
    mK   (aPose.ImVect(Pt3dr(0,0,1)))
{
}


ElMatrix<double> cGpsLoc_Rep::MatRot() const
{
   return MatFromCol(mI,mJ,mK);
}

/* ========================================= */
/*                                           */
/*           cGpsLoc_Som                     */
/*                                           */
/* ========================================= */

class cGpsLoc_Som
{
    public :
        cGpsLoc_Som(const std::string & aName);
        cGpsLoc_Rep  & Sol();
	    Pt3dr        & Gps();
        void Save(cNewO_NameManager *);
    private :
        std::string mName;
        cGpsLoc_Rep mSol;//centre perspective+rotation absolute calculé
        Pt3dr       mGPS;//centre perspectif mesuré par le GPS
};

cGpsLoc_Som::cGpsLoc_Som(const std::string & aName)  :
    mName (aName)
{
}

cGpsLoc_Rep & cGpsLoc_Som::Sol() {return mSol;}
Pt3dr       & cGpsLoc_Som::Gps() {return mGPS;}

void cGpsLoc_Som::Save(cNewO_NameManager * aNM)
{
   CamStenope * aCS = aNM->CalibrationCamera(mName);

   ElRotation3D aR (mSol.Ori(),mSol.MatRot(),true);
   aCS->SetOrientation(aR.inv());

   cOrientationConique anOC =  aCS->StdExportCalibGlob();

   std::string aNameOri = aNM->ICNM()->Assoc1To1("NKS-Assoc-Im2Orient@-"+aNM->OriOut()+"-GpsLoc",mName,true);

   MakeFileXML(anOC,aNameOri);
   std::cout << mName << ", aNameOri=" << aNameOri << "\n";

}

/* ========================================= */
/*                                           */
/*         cGpsLoc_Triplet                   */
/*                                           */
/* ========================================= */

class cGpsLoc_Triplet
{
    public :
        cGpsLoc_Triplet(tGLS_Ptr  aS1,tGLS_Ptr aS2,tGLS_Ptr aS3,const cXml_Ori3ImInit & aXmlOri);
        void InitSomTrivial();
    private :
       tGLS_Ptr     mSoms[3]; //absolute
       cGpsLoc_Rep  mReps[3]; //repere local
       Pt3dr        mPMed;
};

cGpsLoc_Triplet::cGpsLoc_Triplet(tGLS_Ptr  aS1,tGLS_Ptr aS2,tGLS_Ptr aS3,const cXml_Ori3ImInit & aXmlOri) :
    mSoms {aS1,aS2,aS3}
{
    mReps[0] = cGpsLoc_Rep();
    mReps[1] = cGpsLoc_Rep(Xml2El(aXmlOri.Ori2On1()));
    mReps[2] = cGpsLoc_Rep(Xml2El(aXmlOri.Ori3On1()));
    mPMed = aXmlOri.PMed();
}


void cGpsLoc_Triplet::InitSomTrivial()
{

    std::vector<Pt3dr> mVP1; 
    std::vector<Pt3dr> mVP2; 
    
    for (auto aK : {0,1,2})
    {
         //mSoms[aK]->Sol() = mReps[aK];
    
        mVP1.push_back(mReps[aK].Ori());
        mVP2.push_back(mSoms[aK]->Gps());
    }

    //le calcul de la similitude (7parametrs) à partir de points dans les deux reperes 
    cSolBasculeRig    aSol = cSolBasculeRig::StdSolFromPts(mVP1,mVP2);
 
    //la pose calculée
    ElRotation3D      aPose(aSol.Tr(),aSol.Rot(),true);
 
    /*
    std::cout << "Bascule, Tr=" << aSol.Tr() << ", Lambda=" << aSol.Lambda() << "\n";
    std::cout << ", Rot=" << aSol.Rot()(0,0) << ", " << aSol.Rot()(0,1) << ", " << aSol.Rot()(0,2) << "\n"
                          << aSol.Rot()(1,0) << ", " << aSol.Rot()(1,1) << ", " << aSol.Rot()(1,2) << "\n"
                          << aSol.Rot()(2,0) << ", " << aSol.Rot()(2,1) << ", " << aSol.Rot()(2,2) << "\n";
    */
    
    //sauvgaurde de la pose dans l'objet de la classe
    for (auto aK : {0,1,2})
    {
        Pt3dr aPEch( mReps[aK].Ori().x * aSol.Lambda(),
                     mReps[aK].Ori().y * aSol.Lambda(),
                     mReps[aK].Ori().z * aSol.Lambda() );

        //calcul de la position du centre perspectif de la camera aK dans repere absolut 
        Pt3dr              aOriBasc = aPose.ImAff(aPEch); // tk + Mk * Ckj;
        ElMatrix<double>   aRotBasc = aPose.Mat() * mReps[aK].MatRot();// Mk * Mkj

        mSoms[aK]->Sol() = cGpsLoc_Rep(ElRotation3D(aOriBasc,aRotBasc,true));
        
    }

}




/* ========================================= */
/*                                           */
/*         cGpsLoc_Triplet                   */
/*                                           */
/* ========================================= */

class cAppliGpsLoc : public cCommonMartiniAppli
{
     public :
          cAppliGpsLoc(int argc,char ** argv);
     private :
          void InitSom();

          std::string                          mDir;
          // Associe un nom d'image a l'objet C++
          std::map<std::string,cGpsLoc_Som *>  mMapS;
          std::vector<cGpsLoc_Triplet>         mV3;
          int                                  mNbSom;
};

void cAppliGpsLoc::InitSom()
{
   
}

cAppliGpsLoc::cAppliGpsLoc(int argc,char ** argv) :
    cCommonMartiniAppli (),
    mDir                ("./")
{
   int tata;
   std::string aPat;
   std::string aDir;
   std::string aGpsOri;
   
   cInterfChantierNameManipulateur * aICNM; 


   //L'args d'entree
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(aPat,"Pattern of images", eSAM_IsExistFile) 
                   << EAMC(aGpsOri,"GPS orientation (OrientationConique)", eSAM_IsExistFile) ,
        LArgMain() << EAM(tata,"GenOri",true,"Generate Ori, Def=true, false for quick process to RedTieP")
                   << ArgCMA()
   );
 
   #if (ELISE_windows)
        replace( aPat.begin(), aPat.end(), '\\', '/' );
   #endif

   SplitDirAndFile(aDir,aPat,aPat);
   StdCorrecNameOrient(aGpsOri,aDir);//ajoutera "Ori-" devant aGpsOri si necessaire

   //classe qui gerent des fichiers (lecture d'orientation GPS)
   aICNM     = cInterfChantierNameManipulateur::BasicAlloc(aDir);
   aGpsOri   = aICNM->StdKeyOrient(aGpsOri); 

   //recuperation toutes les images coherent avec un pattern
   cElemAppliSetFile anEASF(aPat);
   const std::vector<std::string> * aSetName =   anEASF.SetIm();

   //lecture des sommets, pour chaque image 
   //  + recuper son sommet (aC) d'un fichier d'orientation 
   //  + cree une classe cGpsLoc_Som et  mets la dans la map mMapS
   //  + initialise la pose avec la pose venant de GPS ( mMapS[itL]->Gps() )
   for (const auto & aName : *aSetName)
   {
       std::string aNF = aICNM->Dir() + aICNM->Assoc1To1(aGpsOri,aName,true);

       Pt3dr aC = StdGetObjFromFile<Pt3dr>
                    (
                        aNF,
                        StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                        "Centre",
                        "Pt3dr"
                    );

      // Cree l'objet representant une image et le rentre dans le dictionnaire
      mMapS[aName] = new cGpsLoc_Som(aName);
      mMapS[aName]->Gps() = aC;      

      //std::cout << "+Pt3dr=" << mMapS[aName]->Gps() << "\n"; 

   }


   //encore une classe qui gere les triplets
   cNewO_NameManager *  aNM =NM(mDir);
   std::string aNameLTriplets = aNM->NameTopoTriplet(true);
   //lecture d'un fichier xml contenant une liste de tous les triplets
   cXml_TopoTriplet  aLT = StdGetFromSI(aNameLTriplets,Xml_TopoTriplet);

   //pour chaque triplets 
   //  + verifie s'il existe son sommet
   //  + recuper les cGpsLoc_Som correspondant a chaque image d'un triplet (absolut)
   //  + recuper le nom de fichier qui contient l'orientation d'un triplet (relatif)(aName3R) 
   //  + lecture du fichier (aXml3Ori)
   //  + "push" de la classe cGpsLoc_Triplet dans le vecteur mV3; cGpsLoc_Triplet contient les trois sommets (GPS=absolut) et ses orientations (relative)
   for (auto a3 : aLT.Triplets())
   {
       if (DicBoolFind(mMapS,a3.Name1()) && DicBoolFind(mMapS,a3.Name2()) && DicBoolFind(mMapS,a3.Name3()))
       {
          cGpsLoc_Som * aS1 = mMapS[a3.Name1()];
          cGpsLoc_Som * aS2 = mMapS[a3.Name2()];
          cGpsLoc_Som * aS3 = mMapS[a3.Name3()];

          std::string  aName3R = aNM->NameOriOptimTriplet(true,a3.Name1(),a3.Name2(),a3.Name3());
          cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);
          mV3.push_back(cGpsLoc_Triplet(aS1,aS2,aS3,aXml3Ori));

       }
   }
   mNbSom = mMapS.size();


   // Cas particulier ou il n'y a que 3 sommets
   if (mNbSom==3)
   {
      ELISE_ASSERT(mV3.size()==1,"Incoherent size with on triplet");
      mV3[0].InitSomTrivial();
      for (auto aPair : mMapS)
           aPair.second->Save(aNM);
   }
   else
   {   // calcul pour chaque triplet
/* Commente car warning unused
       for (auto a3 : mV3)
       {
            //a faire:
            //- calcul de lorientation absolute pour chaque sommet dans chaque triplet independement
            //  (alors on dispose de plusiers poses abs pour chaque sommet)
            //- estimer la transformation de similitude (7param) la plus robuste en prennant en compte toutes les resultat 
       }
*/

   }
}


int CPP_NOGpsLoc(int argc,char ** argv)
{
   //l'entree de l'application
   cAppliGpsLoc anAppli(argc,argv);
   return EXIT_SUCCESS;
}





