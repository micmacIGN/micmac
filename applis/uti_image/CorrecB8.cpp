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

class cCorreB8
{
    public :
       cCorreB8(const std::string & aDir,const std::string & aName,INT aBitTest);
       void MakeStat();

       void Corrige();
       void ForceG();
    private :
       void CorrigeShit(INT);
       std::string mDir;
       std::string mName;
       std::string mNameFull;
       Tiff_Im     mTif;
       Pt2di       mSz;
       Im2D_U_INT2 mIm;
       U_INT2 **   mDIm;
       INT         mBitTest;
       INT         mFlagBitTest;
       REAL        mCptMinMont;
       REAL        mCptMaxMont;
       REAL        mCptIntermMont;
       REAL        mCptMinDesc;
       REAL        mCptMaxDesc;
       REAL        mCptIntermDesc;

       bool        mDecGauche;
       bool        mDecDroite;
       bool        mNoDec;
};

void cCorreB8::ForceG()
{
  mDecGauche = true;
  mDecDroite = false;
  mNoDec = false;
}

static const INT TheRabY = 100;
static const INT TheRabX = 100;

cCorreB8::cCorreB8 (const std::string & aDir, const std::string & aName,INT aBitTest) :
   mDir         (aDir),
   mName        (aName),
   mNameFull    (mDir+mName),
   mTif         (Tiff_Im::StdConv(mNameFull)),
   mSz          (mTif.sz()),
   mIm          (mSz.x,mSz.y),
   mDIm         (mIm.data()),
   mBitTest     (aBitTest),
   mFlagBitTest (1<<mBitTest)
{
}

void cCorreB8::CorrigeShit(INT aShift)
{
   if (aShift)
   {

       mFlagBitTest = 2048 | 1024 | 512 | 256;

       Symb_FNum aF0 (    (mIm.in()  &(~mFlagBitTest))
                        | (trans(mIm.in(0)&mFlagBitTest,Pt2di(aShift,0)))
                     );
       // Symb_FNum aF0 ( 513);
      Fonc_Num aFCor = (aF0>>8) | ((aF0&255) << 8);
      ELISE_COPY
      (
         mTif.all_pts(),
         aFCor,
         mTif.out()
      );
   }
}


void cCorreB8::MakeStat()
{
   mCptMaxMont= 0;
   mCptMinMont= 0;
   mCptIntermMont= 0;
   mCptMaxDesc= 0;
   mCptMinDesc= 0;
   mCptIntermDesc= 0;

   ELISE_COPY(mTif.all_pts(),mTif.in(),mIm.out());

   for (INT aY = TheRabY ; aY < mSz.y -TheRabY ; aY++)
   {
       U_INT2 * aLine = mDIm[aY];
       // for (INT aX0 =0, aX1=mSz.x-1 ; aX0<aX1 ; aX0++,aX1--)
       //     ElSwap(aLine[aX0],aLine[aX1]);
       for (INT aX = TheRabX ; aX<mSz.x-TheRabX ; aX++)
       {
           INT aLastFlag =  aLine[aX-1] & mFlagBitTest;
           INT aFlag =  aLine[aX] & mFlagBitTest;
           INT aNextFlag =  aLine[aX+1] & mFlagBitTest;
           if ( (!aLastFlag) && (aFlag))
           {
               if ((aLine[aX]>aLine[aX-1]) && (aLine[aX]>aLine[aX+1]))
                  mCptMaxMont++;
               else if ((aLine[aX]<aLine[aX-1]) && (aLine[aX]<aLine[aX+1]))
                  mCptMinMont++;
               else
                  mCptIntermMont++;
           }
           if ( (!aNextFlag) && (aFlag))
           {
               if ((aLine[aX]>aLine[aX-1]) && (aLine[aX]>aLine[aX+1]))
                  mCptMaxDesc++;
               else if ((aLine[aX]<aLine[aX-1]) && (aLine[aX]<aLine[aX+1]))
                  mCptMinDesc++;
               else
                  mCptIntermDesc++;
           }
       }
   }
   REAL aMont = (mCptMaxMont+mCptMinMont+mCptIntermMont);
   REAL aDesc = (mCptMaxDesc+mCptMinDesc+mCptIntermDesc);
   
    mCptMaxMont *= 100.0/aMont; 
    mCptMinMont *= 100.0/aMont; 
    mCptIntermMont *= 100.0/aMont; 

    mCptMaxDesc *= 100.0/aDesc; 
    mCptMinDesc *= 100.0/aDesc; 
    mCptIntermDesc *= 100.0/aDesc; 

    mDecGauche =    (mCptIntermMont < 10) 
                 && (mCptMaxMont > 50) 
                 && (mCptMaxMont > mCptMinMont + 10)
                 && (mCptIntermDesc > 50)
                 && (mCptIntermDesc > mCptMaxDesc + 10)
                 && (mCptIntermDesc > mCptMinDesc + 20);

    mDecDroite =    (mCptIntermDesc < 10)
                 && (mCptMaxDesc > 50)
                 && (mCptMaxDesc > mCptMinDesc +10)
                 && (mCptIntermMont > 50)
                 && (mCptIntermMont > mCptMaxMont + 10)
                 && (mCptIntermMont > mCptMinMont + 20);

    mNoDec = (mCptIntermMont > 65) && (mCptIntermDesc > 65);
}

void  cCorreB8::Corrige()
{
   cout << "\n";
   std::string aCor;
   if (mNoDec)
   {
      aCor = "Identi";
      CorrigeShit(0);
   }
   else if (mDecGauche)
   {
      aCor = "Gauche";
      CorrigeShit(-1);
   }
   else if (mDecDroite)
   {
      aCor = "Droite";
      CorrigeShit(1);
   }
   else
   {
      aCor = "Ambigu";
      cout << "AMBIGU \n";
      cout << mCptMaxMont << " " << mCptMinMont << " " << mCptIntermMont << "\n";
      cout << mCptMaxDesc << " " << mCptMinDesc << " " << mCptIntermDesc << "\n";
      if (mCptIntermMont < 10)
      {
         if (mCptMaxMont >mCptMaxDesc)
         {
            aCor = "AGauch";
            CorrigeShit(-1);
         }
         else
         {
            aCor = "ADroit";
            CorrigeShit(1);
         }
      }
   }
/*
    mCptMaxMont *= 100.0/aMont; 
    mCptMinMont *= 100.0/aMont; 
    mCptIntermMont *= 100.0/aMont; 

    mCptMaxDesc *= 100.0/aDesc; 
    mCptMinDesc *= 100.0/aDesc; 
    mCptIntermDesc *= 100.0/aDesc; 
*/

   cout << "File " << mName << " -> " << aCor << "\n";

   std::string  aMV = std::string("mv ")
                     + mNameFull
                     +  std::string(" ")
                     + mDir + aCor +  mName;

   cout << aMV << "\n";
   system(aMV.c_str());
}

/*
GAUCHE 
MONT =  Max 68.2044 Min 29.9075 Interm 1.88812
DESC =  Max 25.0186 Min 9.66006 Interm 65.3214
DOITE
      MONT 25.011 9.66506 65.3239
      DESC 68.1913 29.922 1.88671

*/


int main(int argc,char ** argv)
{
     std::string aPat;
     std::string aDir;
     INT         aForceG=0;
     ElInitArgMain
     (
           argc,argv,
           LArgMain() << EAM(aDir) << EAM(aPat) ,
           LArgMain() << EAM(aForceG,"ForceG",true)
    );	
    std::list<std::string> lNameFile = ListFileMatch(aDir,aPat,1,false);

   for 
   (
        std::list<std::string>::iterator itF=lNameFile.begin();
        itF != lNameFile.end();
        itF++
   )
   {
       cout << *itF << "\n";
       cCorreB8 aCB8(aDir,*itF,8);
       aCB8.MakeStat();
       if (aForceG)
          aCB8.ForceG();
       aCB8.Corrige() ;
   }

/*
*/
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
