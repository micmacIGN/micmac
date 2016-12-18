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
#include "Apero.h"

class cVisuPHom
{
     public :
        cVisuPHom(const Pt2dr & aP1,const Pt2dr & aP2,const double & aRes) :
            mRes (aRes),
            mP1  (aP1),
            mP2  (aP2)
        {
        }

        double  mRes;
        Pt2dr   mP1;
        Pt2dr   mP2;
};

class cVisuResidHom
{
      public :
            cVisuResidHom(const std::string & aIm1,cBasicGeomCap3D *,cBasicGeomCap3D *,ElPackHomologue &,const std::string & aPrefOut);
      private :

            std::string NameFile(const std::string & aPost) {return mPrefOut + aPost;}
            void AddPair(const Pt2dr & aP1,const Pt2dr & aP2);
     
            cBasicGeomCap3D *      mCam1;
            cBasicGeomCap3D *      mCam2;
            ElPackHomologue        mPack;
            std::string            mPrefOut;
            double                 mResol;
            Pt2di                  mSzIm1;
            Pt2di                  mSzResol;
            Im2D_U_INT1            mImSsRes;
            cPlyCloud              mPlyC;
            std::vector<double>    mVRes;
            std::vector<cVisuPHom> mVPH;
            std::vector<double>    mVEpi;
            std::vector<Pt2dr>     mVP1;
            bool                   mDoPly;

            // std::vector<double>   DoOneDegree(int aDegre,double aSigma,
};



 

void cVisuResidHom::AddPair(const Pt2dr & aP1,const Pt2dr & aP2)
{
    double aDif = mCam1->EpipolarEcart(aP1,*mCam2,aP2);

    if (mDoPly)
    {
       mPlyC.AddSphere(Pt3di(255,0,0),Pt3dr(aP1.x,aP1.y,aDif*1000),5,3);
    }
    mVRes.push_back(ElAbs(aDif));
    mVPH.push_back(cVisuPHom(aP1,aP2,aDif));
    mVP1.push_back(aP1);
    mVEpi.push_back(aDif);
}

cVisuResidHom::cVisuResidHom
(
      const std::string & aIm1,
      cBasicGeomCap3D * aCam1,
      cBasicGeomCap3D * aCam2,
      ElPackHomologue & aPack,
      const std::string & aPrefOut
) :
   mCam1    (aCam1),
   mCam2    (aCam2),
   mPack    (aPack),
   mPrefOut (aPrefOut),
   mResol   (10.0),
   mSzIm1   (aCam1->SzBasicCapt3D()),
   mSzResol (round_up(Pt2dr(mSzIm1) / mResol)),
   mImSsRes (mSzResol.x,mSzResol.y),
   mDoPly   (1)
{
     ELISE_fp::MkDirSvp(DirOfFile(aPrefOut));
     if (mDoPly) 
     {
         Tiff_Im aTif(aIm1.c_str());
         ELISE_COPY
         (
             mImSsRes.all_pts(),
             Max(0,Min(255,StdFoncChScale(aTif.in_proj(),Pt2dr(0,0),Pt2dr(mResol,mResol)))),
             mImSsRes.out()
         );

     
         for (int  aX =0 ; aX <mSzResol.x ; aX++)
         {
             for (int  aY =0 ; aY <mSzResol.y ; aY++)
             {
                  double aGr = mImSsRes.data()[aY][aX];
                  mPlyC.AddPt(Pt3di(aGr,aGr,aGr),Pt3dr(aX*mResol,aY*mResol,0));
             }
         }
     }


     for (ElPackHomologue::iterator itP=mPack.begin() ; itP!=mPack.end() ; itP++)
     {
         AddPair(itP->P1(),itP->P2());
     }

     if (mDoPly)
     {
        mPlyC.PutFile(aPrefOut+"-Nuage.ply");
     }

     std::sort(mVRes.begin(),mVRes.end());

     FILE * aFp = FopenNN(aPrefOut+"-Stat.txt","w","VisuHom");
/*
     fprintf(aFp,"NbPts= %d\n",int(mVRes.size()));
     fprintf(aFp,"================= PERC  : RESIDU ==================\n");
      
     for (int aK=0 ; aK<=aNbPerc ; aK++)
     {
         double aRes = mVRes[(aK*(mVRes.size()-1))/aNbPerc];
         fprintf(aFp,"Res[%f]=%f\n",(aK*100.0)/aNbPerc,aRes);
     }
*/

     int aNbPerc = 20;
     Polynome2dReal aPol = Polynome2dReal::PolyDegre1(0,0,0);

     for (int aDeg=0 ; aDeg<6 ; aDeg++)
     {
         fprintf(aFp,"================= PERC  : RESIDU ==================\n");
         if (aDeg==0)
              fprintf(aFp,"    Initial     \n");
         else
              fprintf(aFp,"    Degree=%d     \n",aDeg-1);
         std::vector<double> aVEr;
         for (int aKp=0 ; aKp<int(mVP1.size()) ; aKp++)
             aVEr.push_back(ElAbs(mVEpi[aKp]-aPol(mVP1[aKp])));
         std::sort(aVEr.begin(),aVEr.end());

         for (int aK=0 ; aK<=aNbPerc ; aK++)
         {
             double aRes = aVEr[(aK*(aVEr.size()-1))/aNbPerc];
             fprintf(aFp,"Res[%f]=%f\n",(aK*100.0)/aNbPerc,aRes);
         }
         aPol = LeasquarePol2DFit(aDeg,mVP1,mVEpi,aPol,0.75,2.0,0.5);
     }
     fclose(aFp);
}


int VisuResiduHom(int argc,char ** argv)
{
    std::string aIm1,aIm2,Aero,aSetHom;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aIm1,"Name first image")
                    << EAMC(aIm2,"Name first image")
                    << EAMC(Aero,"Orientation", eSAM_IsExistDirOri),
        LArgMain()
                    << EAM(aSetHom,"SH",true,"Set Homologue")
    );

     std::string aDir,aLocIm1;
     SplitDirAndFile(aDir,aLocIm1,aIm1);

     StdCorrecNameOrient(Aero,aDir);
     StdCorrecNameHomol(aSetHom,aDir);

     cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

     cBasicGeomCap3D * aCam1 = aICNM->StdCamGenOfNames(Aero,aIm1);
     cBasicGeomCap3D * aCam2 = aICNM->StdCamGenOfNames(Aero,aIm2);

     std::string aNameH = aICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+aSetHom+"@dat",aIm1,aIm2,true);
      
     ElPackHomologue  aPack = ElPackHomologue::FromFile(aNameH);

     cVisuResidHom aVRH(aIm1,aCam1,aCam2,aPack,"Visu-Res"+aSetHom +"-" + Aero+ "/"+aIm1+aIm2);

     return EXIT_SUCCESS;
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
