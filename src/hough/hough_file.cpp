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
#include "hough_include.h"


/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

// NbReservCompat entier vide ecrits apres le majic
// number pour compatibilite ulterieure


static const INT NbReservCompat = 100;

class ElHoughFromFile : public ElHoughImplem
{
    public :
       ElHoughFromFile
       (
            ELISE_fp & fp,
            Pt2di Sz,
            REAL StepRho,
            REAL StepTeta,
            REAL RabRho,
            REAL RabTeta
       );
	   void PostInit(ELISE_fp & fp);

       static ElHoughFromFile * NewOne(const std::string & name);
    private :
       virtual void clean(){};
       virtual void ElemPixel(tLCel &,Pt2di) {};
};

static const char * Majic = "eLiSe Hough File";


ElHoughFromFile::ElHoughFromFile
(
     ELISE_fp &fp,
     Pt2di Sz,
     REAL StepRho,
     REAL StepTeta,
     REAL RabRho,
     REAL RabTeta
)   :
    ElHoughImplem(Sz,StepRho,StepTeta,RabRho,RabTeta)
{
}

void ElHoughFromFile::PostInit(ELISE_fp &fp)
{
		ElHoughImplem::PostInit();

    {
        INT nbr = fp.read_INT4();
        SetNbRho(nbr);
        INT nbteta = fp.read_INT4();
        ELISE_ASSERT(nbteta==NbTeta(),"Inc in ElHoughFromFile");
    }

    mIRhoMin = fp.read_INT4();
    mIRhoMax = fp.read_INT4();
    mNbCelTot = fp.read_INT4();
    mFactPds  = fp.read_REAL8();

    // mStepRho = Im1D_REAL8(NbTetaTot());
    mStepRho.read_data(fp);
    // mDataSRho = mStepRho.data();


    // mAdrElem = Im2D_INT4(NbX(),NbY());
    // mDataAdE = mAdrElem.data();
    mAdrElem.read_data(fp);

    // mNbElem = Im2D_U_INT2(NbX(),NbY());
    // mDataNbE = mNbElem.data();
    mNbElem.read_data(fp);

    mIndRho = tImIndex(mNbCelTot);
    mDataIRho = mIndRho.data();
    mIndRho.read_data(fp);

    mIndTeta = tImIndex(mNbCelTot);
    mDataITeta = mIndTeta.data();
    mIndTeta.read_data(fp);

    mGetTetaTrivial = fp.read_INT4();

    mPds = Im1D_U_INT1(mNbCelTot);
    mDataPds = mPds.data();
    mPds.read_data(fp);

    mHouhAccul = Im2D_INT4(NbTetaTot(),NbRho());
    mDataHA = mHouhAccul.data();

    mHouhAcculInit = Im2D_INT4(NbTetaTot(),NbRho());
    mDataHAInit = mHouhAcculInit.data();
    mHouhAcculInit.read_data(fp);

    mMarqBCVS =  Im2D_U_INT1 (mNbTetaTot,NbRho(),0);
}


ElHoughFromFile * ElHoughFromFile::NewOne(const std::string & name)
{
   ELISE_fp fp (name.c_str(),ELISE_fp::READ);

   INT l = (INT) strlen(Majic);
   for (int k=0  ; k<l ; k++)
   {
      U_INT1 c = fp.read_U_INT1();
      ELISE_ASSERT(c==Majic[k],"Bad Majic Number in Hough File");
   }

   {
    for (INT k= 0; k < NbReservCompat ; k++)
        fp.read_INT4();
   }

    INT NbX      = fp.read_INT4();
    INT NbY      = fp.read_INT4();
    REAL8 StepRho  = fp.read_REAL8();
    REAL8 StepTeta = fp.read_REAL8();
    REAL8 RabRho = fp.read_REAL8();
    REAL8 RabTeta = fp.read_REAL8();

cout << NbX << " " << NbY << " "
     << StepRho << " "
     << StepTeta << " "
     << RabTeta<< "\n";

    ElHoughFromFile * aRes = new  ElHoughFromFile(fp,Pt2di(NbX,NbY),StepRho,StepTeta,RabRho,RabTeta);
	aRes->PostInit(fp);
	return aRes;
}

ElHough * ElHough::NewOne(const ElSTDNS string & name)
{
   return ElHoughFromFile::NewOne(name);
}


void ElHoughImplem::write_to_file(const std::string & name) const
{
    ELISE_fp fp (name.c_str(),ELISE_fp::WRITE);

    fp.str_write(Majic);

    for (INT k= 0; k < NbReservCompat ; k++)
        fp.write_INT4(0);

    fp.write_INT4(NbX());
    fp.write_INT4(NbY());
    fp.write_REAL8(mStepRhoInit);
    fp.write_REAL8(mStepTeta*LongEstimTeta());
    fp.write_REAL8(mRabRho);
    fp.write_REAL8(mRabTeta);


    fp.write_INT4(NbRho());
    fp.write_INT4(NbTeta());

    fp.write_INT4(mIRhoMin);
    fp.write_INT4(mIRhoMax);
    fp.write_INT4(mNbCelTot);
    fp.write_REAL8(mFactPds);

    mStepRho.write_data(fp);
    mAdrElem.write_data(fp);
    mNbElem.write_data(fp);
    mIndRho.write_data(fp);
    mIndTeta.write_data(fp);
    fp.write_INT4(mGetTetaTrivial);
    mPds.write_data(fp);
    mHouhAcculInit.write_data(fp);

    fp.close();
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
