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
#include "general/all.h"
#include "private/all.h"
#include "XML_GEN/all.h"
#include <algorithm>

using namespace NS_ParamChantierPhotogram;

class cSom;
class cGraphHom;

class cSom
{
     public :
       cSom(const cGraphHom & aGH,const std::string & aName,ElCamera * aCam,Pt3dr aC) :
          mGH   (aGH),
          mName (aName),
          mCam (aCam),
          mC   (aC)
       {
       }

       const cGraphHom & mGH;
       std::string       mName;
       ElCamera *        mCam;
       Pt3dr             mC;

       bool HasInter(const cSom & aS2) const;
};



class cGraphHom
{
    public :
        
        friend class cSom;

        cGraphHom(int argc,char ** argv);
        void DoAll();

    private :

        std::string mDir;
        std::string mPat;
        std::string mKeyFile;
        cInterfChantierNameManipulateur * mICNM;
        
        std::string mOut;
        std::string mTagC;
        std::string mTagOri;

        std::list<std::string>  mLFile;
        std::vector<cSom *>    mVC;
        int                    mNbSom;
        double                 mAltiSol;
        double                 mDist;
        double                 mRab;
        bool                   mTerr;
        bool                   mSym;
};


bool cSom::HasInter(const cSom & aS2) const
{
    if (euclid(mC-aS2.mC) > mGH.mDist)
       return false;

    if ((! mGH.mTerr) && mCam && (aS2.mCam))
    {
         const cElPolygone &  aPol1= mCam->EmpriseSol();
         const cElPolygone &  aPol2= aS2.mCam->EmpriseSol();
         const cElPolygone &  aInter = aPol1 * aPol2;
          
         if (aInter.Surf() <= 0) return false;

    }

    return true;
}




cGraphHom::cGraphHom(int argc,char ** argv) :
      mOut      ("GrapheHom.xml"),
      mTagC     ("Centre"),
      mTagOri   ("OrientationConique"),
      mAltiSol   (0),
      mDist      (-1),
      mRab       (0.2),
      mTerr      (false),
      mSym       (true)
{
    int mVitAff = 10;
      
    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(mDir)
                    << EAM(mPat) 
                    << EAM(mKeyFile),
        LArgMain()  << EAM(mTagC,"TagC",true)
                    << EAM(mTagOri,"TagOri",true)
                    << EAM(mAltiSol,"AltiSol",true)
                    << EAM(mDist,"Dist",true)
                    << EAM(mRab,"Rab",true)
                    << EAM(mTerr,"Terr",true)
                    << EAM(mSym,"Sym",true)
                    << EAM(mOut,"Out",true)
        
    );
    ELISE_ASSERT(mRab>=0,"Rab <0");


    cTplValGesInit<std::string>  aTplFCND;
    mICNM = cInterfChantierNameManipulateur::StdAlloc(argc,argv,mDir,aTplFCND);

    mKeyFile = mICNM->StdKeyOrient(mKeyFile);

    mLFile =  mICNM->StdGetListOfFile(mPat);


    mNbSom =  mLFile.size();

    std::cout << "Nb Images = " <<  mNbSom << "\n";

    int aNbDiam = 0;
    double aSomDiam = 0;
    int aCpt = 0;

    for 
    (
         std::list<std::string>::const_iterator itS=mLFile.begin();
         itS!=mLFile.end();
         itS++
    )
    {
         std::string aNF = mICNM->Dir() + mICNM->Assoc1To1(mKeyFile,*itS,true);

         Pt3dr aC = StdGetObjFromFile<Pt3dr>
                    (
                        aNF,
                        StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                        mTagC,
                        "Pt3dr"
                    );

         cOrientationConique * aCO = OptionalGetObjFromFile_WithLC<cOrientationConique>
                                     (
                                           0,0,
                                           aNF,
                                           StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                           mTagOri,
                                           "OrientationConique"
                                     );

         ElCamera * aCam = 0;
         if (aCO) 
         {
            if (!mTerr)
            {
                cTplValGesInit<double> & aZ = aCO->Externe().AltiSol();
                if (! aZ.IsInit())
                {
                   aZ.SetVal(mAltiSol);
                }

                 double aDZ =  aC.z - aZ.Val();
                 aZ.SetVal(aC.z - aDZ *(1+mRab));
            }

            aCam = Cam_Gen_From_XML(*aCO,mICNM);

            if (! mTerr) 
            {
                double aDiam = aCam->EmpriseSol().DiamSimple();
                aNbDiam ++;
                aSomDiam += aDiam;
            }
         }


         mVC.push_back(new cSom(*this,*itS,aCam,aC));
         if ((aCpt %mVitAff) == (mVitAff-1)) 
         {
            std::cout << "Load  : remain " << (mNbSom-aCpt) << " to do\n";
         }
         aCpt++;
    }

    if ((mDist<0)  && aNbDiam)
        mDist = aSomDiam / aNbDiam;

    ELISE_ASSERT(mDist>0,"Cannot determine dist ");

     std::cout << "DIST = " << mDist << "\n";

    cSauvegardeNamedRel aRel;
    for (int aK1=0 ; aK1<mNbSom ; aK1++)
    {
        for (int aK2=aK1+1 ; aK2<mNbSom ; aK2++)
        {
             if (mVC[aK1]->HasInter(*(mVC[aK2])))
             {
                aRel.Cple().push_back(cCpleString(mVC[aK1]->mName,mVC[aK2]->mName));
                if (mSym)
                {
                   aRel.Cple().push_back(cCpleString(mVC[aK2]->mName,mVC[aK1]->mName));
                }
             }
             // bool  HasI = mVC[aK1]->HasInter(*(mVC[aK2]));
             // std::cout << HasI  << " " << mVC[aK1]->mName << " " <<  mVC[aK2]->mName  << "\n";
        }
        if ((aK1 %mVitAff) == (mVitAff-1)) 
        {
            std::cout << "Graphe : remain " << (mNbSom-aK1) << " to do\n";
        }
    }
    MakeFileXML(aRel,mDir+mOut);
}


void cGraphHom::DoAll()
{
}




int main(int argc,char ** argv)
{
    cGraphHom aGr(argc,argv);
    aGr.DoAll();
}




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
