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


/**
 * MasqMaker: make masks form pictures
 * */

//
//----------------------------------------------------------------------------

int MasqMaker_main(int argc,char ** argv)
{
    std::string aFullPattern;
    int aImMinVal=5000;
    int aImMaxVal=-1000;
    std::string aMasqSup="?";
    int aSzW=0;

    cout<<endl<<"MasqMaker: make masks form pictures."<<endl<<endl;
	ElInitArgMain
		(
		argc, argv,
		//mandatory arguments
        LArgMain()  << EAMC(aFullPattern, "Pattern images",  eSAM_IsPatFile)
                    << EAMC(aImMinVal, "Minimum value")
                    << EAMC(aImMaxVal, "Maximum value"),
		//optional arguments
		LArgMain() << EAM(aMasqSup, "MasqSup", true, "Supplementary mask")
                   << EAM(aSzW, "SzW", true, "SzW for masking (def=0)")
		);

    if (MMVisualMode) return EXIT_SUCCESS;
    
    // Initialize name manipulator & files
    std::string aDirImages,aPatIm;
    SplitDirAndFile(aDirImages,aPatIm,aFullPattern);
    std::cout<<"Working dir: "<<aDirImages<<std::endl;
    std::cout<<"Images pattern: "<<aPatIm<<std::endl;


    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));

    Tiff_Im *aMasqSupTif=0;
    if (aMasqSup!="?")
    {
        aMasqSupTif=new Tiff_Im(aMasqSup.c_str());
    }

    for (unsigned int i=0;i<aSetIm.size();i++)
    {
        std::cout<<"Masking "<<aSetIm[i]<<"..."<<std::flush;
        string aNameMasq=StdPrefix(aSetIm[i]) + "_Masq.tif";
        string NameXML=StdPrefix(aSetIm[i]) + "_Masq.xml";

        std::string aNameImageTif = NameFileStd(aSetIm[i],1,false,true,true,true);
        Tiff_Im aPicTiff(aNameImageTif.c_str());
        TIm2D<U_INT1, INT4> aMasqImageT(aPicTiff.sz());
        Im2D<U_INT1,INT4>  aMasqImage(aMasqImageT._the_im);

        TIm2D<U_INT1,INT4> aOriginImImT(aPicTiff.sz());
        Im2D<U_INT1,INT4>  aOriginImIm(aOriginImImT._the_im);
        ELISE_COPY(aOriginImIm.all_pts(), (aPicTiff.in() ), aOriginImIm.out());

        ELISE_COPY(aMasqImage.all_pts(),
                   (rect_min(aOriginImIm.in(0),aSzW)>aImMinVal)
                   *(rect_max(aOriginImIm.in(0),aSzW)<aImMaxVal)*255,
                   aMasqImage.out());

        Tiff_Im  aFileOut
            (
            aNameMasq.c_str(),
            aPicTiff.sz(),
            GenIm::bits1_msbf,
            //Tiff_Im::Group_4FAX_Compr,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero
            );

        if (aMasqSupTif)
        {
            ELISE_ASSERT(aPicTiff.sz() == aMasqSupTif->sz(), "Picture and MasqSup must have the same size!");

            ELISE_COPY
                (
                aFileOut.all_pts(),
                (aMasqImage.in()*aMasqSupTif->in())>0,
                aFileOut.out()
                );
        }else{
            ELISE_COPY
                (
                aFileOut.all_pts(),
                aMasqImage.in()>0,
                aFileOut.out()
                );
        }

        cFileOriMnt aOriMnt;
        aOriMnt.NameFileMnt()=aNameMasq;
        aOriMnt.NombrePixels()=aPicTiff.sz();
        aOriMnt.OriginePlani()=Pt2dr(0,0);
        aOriMnt.ResolutionPlani()=Pt2dr(1,1);
        aOriMnt.OrigineAlti()=0;
        aOriMnt.ResolutionAlti()=1;
        aOriMnt.Geometrie()=eGeomMNTFaisceauIm1PrCh_Px1D;

        MakeFileXML(aOriMnt,NameXML);

        std::cout<<" done!"<<std::endl;
    }
    if (aMasqSupTif) delete aMasqSupTif;

    std::cout<<"MasqMaker finished.\n";

    return EXIT_SUCCESS;
}

/* Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilite au code source et des droits de copie,
de modification et de redistribution accordes par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
seule une responsabilite restreinte pese sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concedants successifs.

A cet egard  l'attention de l'utilisateur est attiree sur les risques
associes au chargement,  a l'utilisation,  a la modification et/ou au
developpement et a la reproduction du logiciel par l'utilisateur etant
donne sa specificite de logiciel libre, qui peut le rendre complexe a
manipuler et qui le reserve donc a des developpeurs et des professionnels
avertis possedant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invites a charger  et  tester  l'adequation  du
logiciel a leurs besoins dans des conditions permettant d'assurer la
securite de leurs systemes et ou de leurs donnees et, plus generalement,
a l'utiliser et l'exploiter dans les memes conditions de securite.

Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
termes.
Footer-MicMac-eLiSe-25/06/2007/*/
