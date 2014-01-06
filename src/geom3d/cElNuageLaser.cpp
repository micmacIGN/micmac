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


#include "StdAfx.h"
#include "algo_geom/qdt.h"


class 	cTypeP3toP2
{
    public :
        Pt2dr operator () (const Pt3dr & aP)
        {
           return Pt2dr(aP.x,aP.y);
        }
};


class cQtcElNuageLaser : public ElQT<Pt3dr,Pt2dr,cTypeP3toP2>
{
      public :
         cQtcElNuageLaser
         (
             Box2dr        aBox,
             INT           aNbObjMax,
             REAL          aSzMin
         )  :
            ElQT<Pt3dr,Pt2dr,cTypeP3toP2>(cTypeP3toP2(),aBox,aNbObjMax,aSzMin)
         {
         }
};

/*************************************************/
/*                                               */
/*                cElNuageLaser                  */
/*                                               */
/*************************************************/


cElNuageLaser::cElNuageLaser
(
     const std::string & aNameFile,
     const char *  aNameOri,
     const char *  aNameGeomCible ,
     const char *  aNameGeomInit

)  :
   mVPts (),
   mQt   (0)
{
   std::string aNameBin = StdPrefix(aNameFile) + ".tif";
   if (! ELISE_fp::exist_file(aNameBin))
   {
       INT aNb = 3;
       FILE * aFP = ElFopen(aNameFile.c_str(),"r");
       ELISE_ASSERT(aFP!=0,"Cannot Open File for Laser Data");

       char Buf[10000];
       INT aCpt =0;
       while (aNb>=3)
       {
           aNb=0;
           char * got = fgets(Buf,10000,aFP);
           Pt3dr aP;
           if (got)
           {
              aNb = sscanf(Buf,"%lf %lf %lf",&aP.x,&aP.y,&aP.z);
           }
	   if (aNb>=3)
              mVPts.push_back(aP);
	   aCpt++;
       }

       INT aNbPts = (INT) mVPts.size();
       Im2D_REAL8 aImPts(aNbPts,3);
       REAL ** aData = aImPts.data();
       for (INT aK=0 ; aK<aNbPts ; aK++)
       {
	   Pt3dr aP = mVPts[aK];
           aData[0][aK] = aP.x;
           aData[1][aK] = aP.y;
           aData[2][aK] = aP.z;
       }

       Tiff_Im aFile
	       (
                  aNameBin.c_str(),
		  Pt2di(aNbPts,3),
		  GenIm::real8,
		  Tiff_Im::No_Compr,
		  Tiff_Im::BlackIsZero,
		     Tiff_Im::Empty_ARG
		  +  Arg_Tiff(Tiff_Im::ANoStrip())
               );
       ELISE_COPY(aImPts.all_pts(),aImPts.in(),aFile.out());
   }
   else
   {
       Tiff_Im aFile(aNameBin.c_str());
       Pt2di aSz = aFile.sz();
       Im2D_REAL8 aImPts(aSz.x,aSz.y);
       ELISE_COPY(aImPts.all_pts(),aFile.in(),aImPts.out());

       REAL ** aD = aImPts.data();
       mVPts.reserve(aSz.x);
       for (INT aK=0 ; aK<aSz.x ; aK++)
       {
           Pt3dr aP(aD[0][aK],aD[1][aK],aD[2][aK]);
           mVPts.push_back(aP);
       }
   }


   Ori3D_Std * aOri = 0;
   eModeConvGeom aMode = eConvId;

   if (aNameOri)
   {
       if (!strcmp(aNameGeomInit,"GeomCarto"))
       {
           if (!strcmp(aNameGeomCible,"GeomCarto"))
           {
               aMode = eConvId;
           }
           else if (!strcmp(aNameGeomCible,"GeomTerrain"))
           {
               aMode = eConvCarto2Terr;
           }
           else if (!strcmp(aNameGeomCible,"GeomTerIm1"))
           {
               aMode = eConvCarto2TerIm;
           }
           else
           {
              ELISE_ASSERT(false,"Bad GeomCible in cElNuageLaser::cElNuageLaser");
           }
       }
       else
       {
           ELISE_ASSERT(false,"Bad GeomInit in cElNuageLaser::cElNuageLaser");
       }
       if (aMode != eConvId)
          aOri = new Ori3D_Std (aNameOri) ;
   }

   for (INT aK=0 ; aK<INT( mVPts.size()); aK++)
   {
       Pt3dr aP = mVPts[aK];
       if (aOri)
       {
          if (aMode == eConvCarto2Terr)
             aP = aOri->carte_to_terr(aP);
          else if (aMode == eConvCarto2TerIm)
          {
             aP = aOri->carte_to_terr(aP);
             Pt2dr aP2 = aOri->to_photo(aP);
             aP.x = aP2.x;
             aP.y = aP2.y;
          }
          mVPts[aK] = aP;
       }

       REAL  aZ   =  aP.z;
       Pt2dr aP2 (aP.x,aP.y);
       if (aK==0)
       {
           mZMax = mZMin = aZ;
           mPInf =mPSup = aP2;
       }
       ElSetMin(mZMin,aZ);
       ElSetMax(mZMax,aZ);
       mPInf.SetInf(aP2);
       mPSup.SetSup(aP2);
   }
   delete aOri;
}


void cElNuageLaser::AddQt(INT aNbObjMaxParFeuille,REAL aDistPave)
{
   delete mQt;

   REAL aD = euclid(mPInf,mPSup) * 1e-4;
   Pt2dr aPRab(aD,aD);

   mQt = new  cQtcElNuageLaser
              (
                    Box2dr(mPInf-aPRab,mPSup+aPRab),
                    aNbObjMaxParFeuille,
                    aDistPave
              );

   for (INT aK=0 ; aK<INT( mVPts.size()); aK++)
   {
      mQt->insert(mVPts[aK]);
   }
}



    // ACCESSEURS

const std::vector<Pt3dr> & cElNuageLaser::VPts() const { return mVPts; }
REAL cElNuageLaser::ZMin()  const {return mZMin;}
REAL cElNuageLaser::ZMax()  const {return mZMax;}
Box2dr cElNuageLaser::Box() const {return Box2dr(mPInf,mPSup);}


class cENL_cTplResRVoisin : public cTplResRVoisin<Pt3dr>
{
    public :
       cENL_cTplResRVoisin(cResReqNuageLaser & aRes) : mRes (aRes) {}
  virtual ~cENL_cTplResRVoisin() {}
    private :
       void Add(const Pt3dr & aP) {mRes.cResReqNuageLaser_Add(aP);}
       cResReqNuageLaser & mRes;
};

void cElNuageLaser::ParseNuage(cResReqNuageLaser & aResNuage,Box2dr aBox)
{
   ELISE_ASSERT(mQt!=0,"No Qt for cElNuageLaser::ParseNuage");

    cENL_cTplResRVoisin aResGen(aResNuage);
    mQt->RVoisins(aResGen,aBox,1e-5);
}

/*

void cElNuageLaser::SauvCur(const std::string &)
{
  INT aNbP = mVPts.size();
  Im2D_REAL8 aImPts(aNbP,3);

  for (INT aK=0; aK<aNbP ; aK++)
  {
      Pt3dr aP1   =  mVPts[aK];
      REAL anAbc = mAppli.AbcisseVol(aP1);
      ElRotation3D aRot = mRotPts->CurRot(anAbc);
      Pt3dr aP2 = aRot.ImAff(aP2);
      if (aK%5000 == 0)
         cout << aK << " " << aP1-aP2 << euclid(aP1-aP2) << "\n";;
  }
}
*/






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
