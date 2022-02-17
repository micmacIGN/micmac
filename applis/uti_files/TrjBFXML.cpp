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


using namespace NS_ParamChantierPhotogram;

class cLinkTrj;
class cTrjPMul;
class cImTrjBF2Xml;
class cTrjBF2Xml;


/************************************************************/
/*                                                          */
/*                      cLinkTrj                            */
/*                                                          */
/************************************************************/

class cLinkTrj
{
     public :
     private  :
};

/************************************************************/
/*                                                          */
/*                      cTrjPMul                            */
/*                                                          */
/************************************************************/

class cTrjPMul
{
    public : 

       cTrjPMul(const std::string & aName) ;
       void AddPt(cImTrjBF2Xml *,const Pt2dr &);
       void SetTer(const Pt3dr & aPTer);

       void DoHom( cTrjBF2Xml&);
    private : 
       std::string                  mName;
       bool                         mHasTer;
       Pt3dr                        mTer;
       std::vector<cImTrjBF2Xml *>  mIms;
       std::vector<Pt2dr>           mPts;
};

cTrjPMul::cTrjPMul(const std::string & aName) :
   mName   (aName) ,
   mHasTer (false)
{
}

void cTrjPMul::AddPt(cImTrjBF2Xml * aIm,const Pt2dr & aPt)
{
   mIms.push_back(aIm);
   mPts.push_back(aPt);
   //  AddPt(aIm,aPt);
}

void cTrjPMul::SetTer(const Pt3dr & aPTer)
{
   ELISE_ASSERT(!mHasTer,"Multiple cTrjPMul::SetTer");
   mHasTer = true;
   mTer = aPTer;
}


/************************************************************/
/*                                                          */
/*                    cImTrjBF2Xml                          */
/*                                                          */
/************************************************************/

class cImTrjBF2Xml
{
     public :
         cImTrjBF2Xml(const std::string & aName,const std::string & aKey,const std::string& aNumB) ;
         void  SetGPS(const Pt3dr & aGPS);
         void Chaine(cImTrjBF2Xml * aNext);

         void CalcOrientationInit();
         const std::string &  Key() const {return mKey;}
         const std::string &  Name() const {return mName;}
         const std::string &  NumB() const {return mNumB;}

         cOrientationConique OC();
         const Pt3dr & GPS() const ;
         bool  HasGPS() const {return mHasGPS;}


     private :
         cImTrjBF2Xml * NorThis() {return mNext ? mNext : this;}
         cImTrjBF2Xml * PorThis() {return mPrec ? mPrec : this;}
    

         std::string  mName;
         std::string  mKey;
         std::string  mNumB;
         bool         mHasGPS;
         Pt3dr        mGPS;
         ElRotation3D mOri;
         
         cImTrjBF2Xml * mNext;
         cImTrjBF2Xml * mPrec;
};

const Pt3dr & cImTrjBF2Xml::GPS() const 
{
   ELISE_ASSERT(mHasGPS,"cImTrjBF2Xml::GPS");
   return mGPS;
}

void cImTrjBF2Xml::CalcOrientationInit()
{
     Pt3dr aZ(0,0,-1);
     
     cImTrjBF2Xml * aN = NorThis();
     cImTrjBF2Xml * aP = PorThis();

     ELISE_ASSERT(aN!=aP,"cImTrjBF2Xml::CalcOrientationInit");

     Pt3dr aY = -vunit(aN->GPS() - aP->GPS());

     Pt3dr aX = aY ^ aZ;

     mOri = ElRotation3D
            (
                 GPS(),
                 ElMatrix<double>::Rotation(aX,aY,aZ)
            );

}


cOrientationConique cImTrjBF2Xml::OC()
{
   cOrientationConique aRes;

   aRes.FileInterne().SetVal("toto.xml");
   aRes.ConvOri().KnownConv().SetVal(eConvApero_DistM2C);
   aRes.Externe() = From_Std_RAff_C2M(mOri,true);
 
   return aRes;
}

void  cImTrjBF2Xml::SetGPS(const Pt3dr & aGPS)
{
   if (mHasGPS)
   {
      std::cout << "For name " << mName << "\n";
      ELISE_ASSERT(false,"Multiple GPS");
   }
   mHasGPS = true;
   mGPS = aGPS;
}

void cImTrjBF2Xml::Chaine(cImTrjBF2Xml * aNext)
{
   if (mNumB == aNext->mNumB)
   {
       mNext = aNext;
       aNext->mPrec = this;
   }
}


cImTrjBF2Xml::cImTrjBF2Xml(const std::string & aName,const std::string & aKey,const std::string& aNumB) :
   mName    (aName),
   mKey     (aKey),
   mNumB    (aNumB),
   mHasGPS  (false),
   mOri     (ElRotation3D::Id),
   mNext    (0),
   mPrec    (0)
{
}




/************************************************************/
/*                                                          */
/*                    cTrjBF2Xml                            */
/*                                                          */
/************************************************************/

#define SzBuf 10000
static char Buf[SzBuf];

class cTrjBF2Xml
{
    public :
        cTrjBF2Xml (int argc,char **argv,const std::string & aDir,const std::string & aPref);
        

        void AddGPS(const std::string&);
        void AddPtsImage(const std::string&);
        void AddPtsTer(const std::string&);


        void CalcOrientationInit();
        void DoHom();

        ElPackHomologue & PackOfKeys(const std::string&,const std::string&);



        void SauvHoms();
        void SauvIms();
    private :



        cImTrjBF2Xml * NewImOfName(const std::string & aKey,const std::string & aNameBande);
        cImTrjBF2Xml * ExistingImOfName(const std::string & aKey);



        cTrjPMul *     PtOfName(const std::string & aKey,bool CanCreate);
        FILE * StdF(const std::string & aExt);
        std::string  StdNF(const std::string & aExt);

        cInterfChantierNameManipulateur *     mICNM;
        std::string                           mDir;
        std::string                           mPref;
        std::string                           mExtGPS;
        std::map<std::string,cImTrjBF2Xml *>  mIms;
        std::map<std::string,cTrjPMul *>      mPts;
        cImTrjBF2Xml *                        mLastIm;

        std::map<std::pair<std::string,std::string> ,ElPackHomologue> mHoms;
};


ElPackHomologue & cTrjBF2Xml::PackOfKeys(const std::string& aK1,const std::string& aK2)
{
   std::pair<std::string,std::string> aPair;
   aPair.first = aK1;
   aPair.second = aK2;
   return mHoms[aPair];
}

FILE * cTrjBF2Xml::StdF(const std::string & aExt)
{
       return FopenNN(StdNF(aExt),"r","cTrjBF2Xml::StdF");
}

std::string  cTrjBF2Xml::StdNF(const std::string & aExt)
{
       return mDir+mPref+aExt;
}


const cTplValGesInit<std::string>  aArgIMCN;

cTrjBF2Xml::cTrjBF2Xml(int argc,char **argv,const std::string & aDir,const std::string & aPref) :
    mICNM  (cInterfChantierNameManipulateur::StdAlloc(argc,argv,aDir,aArgIMCN)),
    mDir   (aDir),
    mPref  (aPref),
    mLastIm (0)
{
}

cImTrjBF2Xml * cTrjBF2Xml::NewImOfName(const std::string & aKey,const std::string & aNameBande)
{
   cImTrjBF2Xml * & aImP = mIms[aKey];
   ELISE_ASSERT(aImP==0,"cTrjBF2Xml::NewImOfName");
   aImP = new cImTrjBF2Xml(aKey + "_" + aNameBande + ".tif",aKey,aNameBande);

   if (mLastIm)
      mLastIm->Chaine(aImP);

   mLastIm = aImP;
   return aImP;
}

cImTrjBF2Xml * cTrjBF2Xml::ExistingImOfName(const std::string & aKey)
{
   cImTrjBF2Xml * & aImP = mIms[aKey];
   ELISE_ASSERT(aImP!=0,"cTrjBF2Xml::NewImOfName");

   return aImP;
}




cTrjPMul *     cTrjBF2Xml::PtOfName(const std::string & aKey,bool CanCreate)
{
   cTrjPMul * & aPMul = mPts[aKey];
   if (aPMul==0)
   {
      if (!CanCreate)
      {
         std::cout << aKey << "\n";
         ELISE_ASSERT(false,"Cannot create new pt");
      }
      aPMul = new cTrjPMul(aKey);
   }
   return aPMul;
}



void cTrjBF2Xml::AddPtsImage(const std::string& aExt)
{
   FILE * aFp = StdF(aExt);

   while (fgets(Buf,SzBuf,aFp))
   {
        if (Buf[0] != '#')
        {
           int IdPt,IdIm;
           double i,j;
           int aNb = sscanf(Buf,"%d %d %lf %lf",&IdPt,&IdIm,&i,&j);
           ELISE_ASSERT(aNb==4,"cTrjBF2Xml::AddPtsImage");

           
           if (1)
           {
              static int aCpt=0; aCpt++;
              double aPerturb = 3e-2;
              int    aPer = 2000;
              double  aKx = aCpt % aPer;
              double  aKy = aCpt / aPer;
              double aDx =  -aPerturb /2.0 + (aPerturb*aKx) /aPer;
              double aDy =  -aPerturb /2.0 + (aPerturb*aKy) /aPer;

// std::cout << aDx << aDy << "\n";
 
              i += aDx;
              j += aDy;
           }
       

           std::string aKeyIm = ToString(IdIm);
           std::string aKeyPt = ToString(IdPt);

           cImTrjBF2Xml * anIm = ExistingImOfName(aKeyIm);
           cTrjPMul *     aPMul =  PtOfName(aKeyPt,true);

	   aPMul->AddPt(anIm,Pt2dr(i,j));
        }

        // std::cout << aNameFin << "\n";
   }


   ElFclose(aFp);
}


void cTrjBF2Xml::AddPtsTer(const std::string& aExt)
{
   FILE * aFp = StdF(aExt);

   while (fgets(Buf,SzBuf,aFp))
   {
        int IdPt;
        double x,y,z;
        int aNb = sscanf(Buf,"%d %lf %lf %lf",&IdPt,&x,&y,&z);
        ELISE_ASSERT(aNb==4,"cTrjBF2Xml::AddGPS");
       
        std::string aKeyPt = ToString(IdPt);
        cTrjPMul *     aPMul =  PtOfName(aKeyPt,false);

        std::cout << aKeyPt << "\n";
        aPMul->SetTer(Pt3dr(x,y,z));
   }

   ElFclose(aFp);
}

void cTrjBF2Xml::AddGPS(const std::string& aExt)
{
   mExtGPS = aExt;
   FILE * aFp = StdF(aExt);

   while (fgets(Buf,SzBuf,aFp))
   {
        int Id,NumB;
        double x,y,z;
        int aNb = sscanf(Buf,"%d  %d %lf %lf %lf",&Id,&NumB,&x,&y,&z);
        ELISE_ASSERT(aNb==5,"cTrjBF2Xml::AddGPS");
       

        cImTrjBF2Xml * anIm = NewImOfName(ToString(Id),ToString(NumB));
        anIm->SetGPS(Pt3dr(x,y,z));

   }


   ElFclose(aFp);
}

void cTrjBF2Xml::CalcOrientationInit()
{
   for 
   (
        std::map<std::string,cImTrjBF2Xml *>::iterator it=mIms.begin();
        it!=mIms.end();
        it++
   )
   {
       it->second->CalcOrientationInit();
   }
}

void cTrjBF2Xml::DoHom()
{
   for 
   (
        std::map<std::string,cTrjPMul *>::iterator it=mPts.begin();
        it!=mPts.end();
        it++
   )
   {
       it->second->DoHom(*this);
   }
}


void cTrjBF2Xml::SauvHoms()
{
   for
   (
        std::map<std::pair<std::string,std::string> ,ElPackHomologue>::iterator it=mHoms.begin();
        it!=mHoms.end();
        it++
   )
   {
        std::string aName = mICNM->Assoc1To2
                            (
                               "Key-Assoc-CpleIm2HomolPastisBin",
                               ExistingImOfName(it->first.first)->Name(),
                               ExistingImOfName(it->first.second)->Name(),
                               true
                            );
        it->second.StdPutInFile(mDir+aName);
   }
}



void cTrjBF2Xml::SauvIms()
{

   cFichier_Trajecto aFT;
   aFT.NameInit() = StdNF(StdNF(mExtGPS)) ;
   aFT.Lambda() = 1.0;
   aFT.Orient() =   From_Std_RAff_C2M(ElRotation3D::Id,true);
   int aKTime = 0;

   for
   (
        std::map<std::string,cImTrjBF2Xml *>::iterator it=mIms.begin();
        it!=mIms.end();
        it++
   )
   {
       cImTrjBF2Xml * anIm = it->second;
       std::string aNameIm = anIm->Name();
       FILE * aFp= FopenNN(mDir+aNameIm,"w","cTrjBF2Xml::SauvIms");
       ElFclose(aFp);
       cOrientationConique anOC = anIm->OC();
       std::string aNameOr = mICNM->Assoc1To1
                            (
                                "Key-Assoc-Im2OrInit",
                                aNameIm,
                                true
                            );

        MakeFileXML(anOC,mDir+aNameOr,"ExportBF2APERO");

        if (anIm->HasGPS())
        {
           cPtTrajecto aPtT;
           aPtT.Pt() =  anIm->GPS();
           aPtT.IdImage() = anIm->Name();
           aPtT.IdBande() = anIm->NumB();
           aPtT.Time() = aKTime++ ;
           aFT.PtTrajecto()[ aPtT.IdImage()] = aPtT;
        }
   }
   MakeFileXML(aFT,StdPrefix(StdNF(mExtGPS))+".xml","ExportBF2APERO");
}

/************************************************************/
/*                                                          */
/*                      cTrjPMul                            */
/*                                                          */
/************************************************************/

void cTrjPMul::DoHom(cTrjBF2Xml& aTrj)
{
    // std::cout <<  "DoHom " << mIms.size() << "\n";

    for (int aK1=0 ; aK1<int(mIms.size()) ; aK1++)
    {
       for (int aK2=0 ; aK2<int(mIms.size()) ; aK2++)
       {
           if (aK1!=aK2)
           {
                ElPackHomologue & aPack = aTrj.PackOfKeys(mIms[aK1]->Key(),mIms[aK2]->Key());
                ElCplePtsHomologues aCple(mPts[aK1],mPts[aK2]);
// std::cout << mPts[aK1] << (aCpl2.P1()-round_ni(mPts[aK1]))
                aPack.Cple_Add(aCple);
           }
       }
    }
}
//ElPackHomologue & cTrjBF2Xml::PackOfKeys(const std::string& aK1,const std::string& aK2)

/************************************************************/
/*                                                          */
/*                    main                                  */
/*                                                          */
/************************************************************/

int main(int argc,char ** argv)
{
   std::string aNameIn;
   std::string aPostAp = "_appui.txt";
   std::string aPostLiai = "_PtsImage.txt";
   std::string aPostGPS = "_TrajectoGPS.txt";

   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAM(aNameIn) ,
        LArgMain()  << EAM(aPostAp,"Ap",true)
                    << EAM(aPostLiai,"Liais",true)
                    << EAM(aPostGPS,"GPS",true)
   );

   std::string aDir;
   std::string aPref;
   SplitDirAndFile(aDir,aPref,aNameIn);


   cTrjBF2Xml aTrj(argc,argv,aDir,aPref);
  
   aTrj.AddGPS(aPostGPS);
   aTrj.AddPtsImage(aPostLiai);
   aTrj.AddPtsTer(aPostAp);


   aTrj.CalcOrientationInit();
   aTrj.DoHom();


    aTrj.SauvIms();
    aTrj.SauvHoms();


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
