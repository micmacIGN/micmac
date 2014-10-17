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
#include "im_tpl/image.h"




#define DEF_OFSET -12349876

class cAppliTestInterp;

template <class Type> class cTplTestInterp
{
    public :
       typedef typename El_CTypeTraits<Type>::tBase tBase;

       cTplTestInterp(cAppliTestInterp &);
       std::string  NameSauv(const std::string &);

       void TestInterpol(const std::string & aName,cInterpolateurIm2D<Type> * anInterp);

       cAppliTestInterp & mAppli;
       Pt2di              mSz;

       Im2D<Type,tBase>   mImIn;
       TIm2D<Type,tBase>  mTImIn;

       Im2D<Type,tBase>   mImOut;
       TIm2D<Type,tBase>  mTImOut;

};

class  cAppliTestInterp
{
    public :
        cAppliTestInterp(const std::string & aName,int argc,char ** argv);

      

        std::string         mNameIn;
        std::string         mDir;
        Tiff_Im             mTifIn;
        GenIm::type_el      mTypeIn;
        Pt2di               mSz;
        double              mEch;
};


    //==========================================
    //==========================================
    //==========================================


template <class Type> 
cTplTestInterp<Type>::cTplTestInterp(cAppliTestInterp & anAppli) :
   mAppli   (anAppli),
   mSz      (mAppli.mSz),
   mImIn    (mSz.x,mSz.y),
   mTImIn   (mImIn),
   mImOut   (mSz.x,mSz.y),
   mTImOut  (mImOut)
{
     ELISE_COPY
     (
        mAppli.mTifIn.all_pts(),
        mAppli.mTifIn.in(),
        mImIn.out()
     );

    TestInterpol("Bilin",new cInterpolBilineaire<Type>);
    TestInterpol("BiCub",new cInterpolBicubique<Type>(-0.5));
    TestInterpol("TabBiCub",new cTplCIKTabul<Type,tBase>(10,8,-0.5));
    TestInterpol("MPD",new cTplCIKTabul<Type,tBase>(10,8,-0.5,eTabulMPD_EcartMoyen));
}

template <class Type> std::string  
cTplTestInterp<Type>::NameSauv(const std::string & aName)
{
   return     mAppli.mDir 
           + "TestInterp_" 
           + aName + "_" 
           + NameWithoutDir(mAppli.mNameIn);
}


template <class Type> 
void cTplTestInterp<Type>::TestInterpol(const std::string & aName,cInterpolateurIm2D<Type> * anInterp)
{
   Pt2di aPOut;
   int aSzK = anInterp->SzKernel();
   Pt2di aPK(aSzK,aSzK);
   Box2dr aBox(aPK,mSz-aPK-Pt2di(1,1));

   Type** aDIn = mImIn.data();

   for (aPOut.x =0 ; aPOut.x < mSz.x ; aPOut.x++)
   {
       for (aPOut.y =0 ; aPOut.y < mSz.y ; aPOut.y++)
       {
           Pt2dr aPIn = aPOut * mAppli.mEch;
           if (aBox.inside(aPIn))
           {
                mTImOut.oset(aPOut,El_CTypeTraits<Type>::Tronque(anInterp->GetVal(aDIn,aPIn)));
           }
           else
           {
                mTImOut.oset(aPOut,0);
           }
       }
   }
   Tiff_Im::CreateFromIm(mImOut,NameSauv(aName));
}

    //==========================================


cAppliTestInterp::cAppliTestInterp (const std::string & aName, int argc,char ** argv) :
    mNameIn ((aName=="-help") ? "data/TDM.tif" : aName),
    mDir    (DirOfFile(mNameIn)),
    mTifIn  (Tiff_Im::StdConv(mNameIn)),
    mTypeIn (mTifIn.type_el()),
    mSz     (mTifIn.sz()),
    mEch    (0.781)

{
   double aToto;
   ElInitArgMain
   (
	argc,argv,
	LArgMain()  << EAM(mNameIn),
	LArgMain()  << EAM(aToto,"Toto",true)
   );	

   if (mTypeIn==GenIm::u_int1)
   {
        new cTplTestInterp<U_INT1>(*this);
        return;
   }
   else
   {
        ELISE_ASSERT(false,"Type non gere");
   }
}


    //==========================================

int main(int argc,char ** argv)
{
   ELISE_ASSERT(argc>=2,"Pas assez d'arg");
   new cAppliTestInterp (argv[1],argc,argv);
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
