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



#define DEF_OFSET -12349876


int ConvertIm_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);
    Tiff_Im::SetDefTileFile(1000000);


    std::string aNameIn ;

    INT aReducX=0;
    INT aReducY=0;
    INT aReducXY=0;
    INT aVisu=0;
    GenIm::type_el aTypeOut ;
    std::string aNameTypeOut ="";

    Tiff_Im::PH_INTER_TYPE aPhInterpOut ;
    std::string aNamePITOut ="";
    std::string PITOut[] = {"RGB","BW"};
    std::list<std::string> lOut(PITOut, PITOut + sizeof(PITOut) / sizeof(std::string) );

    std::string aNameOut;
    std::string anExt;

    Pt2di aP0(0,0);

    Pt2di aSzOut ;
    Pt2di aSzTF(-1,-1);

    REAL aDyn=1.0;

    Pt2di aSzTileInterne(-1,-1);
    int aKCh = -1;


    std::vector<int> aVPermut;
    int aNoTile = 0;
    std::string aF2 ="";


    ElInitArgMain
    (
    argc,argv,
                LArgMain()  << EAMC(aNameIn, "Image", eSAM_IsExistFile),
    LArgMain()  << EAM(aNameOut,"Out",true)
                << EAM(anExt,"Ext",true)
                    << EAM(aSzOut,"SzOut",true, "Size out", eSAM_NoInit)
                    << EAM(aP0,"P0",true)
                    << EAM(aNameTypeOut,"Type",true, "TypeMNT", eSAM_None, ListOfVal(GenIm::bits1_msbf, ""))
                    << EAM(aNamePITOut,"Col",true, "Col in RGB BW", eSAM_None,lOut)
                    << EAM(aReducXY,"ReducXY",true)
                    << EAM(aReducX,"ReducX",true)
                    << EAM(aReducY,"ReducY",true)
                    << EAM(aVisu,"Visu",true)
                    << EAM(aSzTF,"SzTifTile",true)
                    << EAM(aSzTileInterne,"SzTileInterne",true)
                    << EAM(aDyn,"Dyn",true)
                    << EAM(aKCh,"KCh",true)
                    << EAM(aNoTile,"NoTile",true)
                    << EAM(aVPermut,"Permut",true, "Permut", eSAM_NoInit)
                    << EAM(aF2,"F2",true)
    );

    if (!MMVisualMode)
    {
        // Tiff_Im aTifIn = Tiff_Im::BasicConvStd(aNameIn);
        Tiff_Im aTifIn = Tiff_Im::UnivConvStd(aNameIn);
        INT aNbChIn = aTifIn.nb_chan();

        if (! EAMIsInit(&aTypeOut)) aTypeOut =aTifIn.type_el();
        if (! EAMIsInit(&aPhInterpOut)) aPhInterpOut =  aTifIn.phot_interp();
        if (! EAMIsInit(&aSzOut)) aSzOut = aTifIn.sz();

        if (aReducXY)
        {
            aReducX = 1;
            aReducY = 1;
        }
        if (aNameOut=="")
        {
            if (anExt=="")
            {
                if (aReducX && aReducY)
                    anExt = "_RXY";
                else if (aReducX)
                    anExt = "_RX";
                else if (aReducY)
                    anExt = "_RY";
                else
                    anExt= "_Out";
            }
            if (IsPostfixed(aNameIn))
              aNameOut = StdPrefix(aNameIn) + anExt +"." + StdPostfix(aNameIn);
           else
              aNameOut = aNameIn + anExt + "tif";
        }

        Pt2di aCoefReduc(aReducX != 0 ? 2 : 1, aReducY != 0 ? 2 : 1);
        aSzOut = aSzOut.dcbyc(aCoefReduc);

        if (aNameTypeOut != "")
           aTypeOut = type_im(aNameTypeOut);

        if (aKCh != -1)
           aNamePITOut="BW";

        if ( aVPermut.size() !=0)
        {
             if ( aVPermut.size() ==1)
                 aPhInterpOut = Tiff_Im::BlackIsZero;
             else if ( aVPermut.size() ==3)
                 aPhInterpOut = Tiff_Im::RGB;
             else
            {
               ELISE_ASSERT(aNamePITOut=="","Nb Canaux incoherents");
            }
        }
        else
        {
            if (aNamePITOut=="RGB")
               aPhInterpOut = Tiff_Im::RGB;
            else if (aNamePITOut=="BW")
               aPhInterpOut = Tiff_Im::BlackIsZero;
            else
            {
               ELISE_ASSERT(aNamePITOut=="","Mode Couleur Inconnu");
            }
        }


        Tiff_Im::COMPR_TYPE aComprOut = Tiff_Im::No_Compr;


        L_Arg_Opt_Tiff aLArg = Tiff_Im::Empty_ARG;


        if (! aNoTile)
        {
           if (aSzTileInterne != Pt2di(-1,-1))
               aLArg = aLArg + Arg_Tiff(Tiff_Im::ATiles(aSzTileInterne));

           if (aSzTF != Pt2di(-1,-1))
               aLArg = aLArg + Arg_Tiff(Tiff_Im::AFileTiling(aSzTF));
        }
        else
        {
             aLArg = aLArg + Arg_Tiff(Tiff_Im::ANoStrip());
             aLArg = aLArg + Arg_Tiff(Tiff_Im::AFileTiling(Pt2di(-1,-1)));
        }


        Tiff_Im aTifOut
                (
                      aNameOut.c_str(),
                      aSzOut,
                      aTypeOut,
                      aComprOut,
                      aPhInterpOut,
                      aLArg
                );
        INT aNbChOut = aTifOut.nb_chan();

        Pt2di aSzROut = aSzOut;
        Output anOut = aTifOut.out();

        Fonc_Num aFin = aTifIn.in_proj();
        if (aF2!="")
        {
             Tiff_Im aT2 = Tiff_Im::BasicConvStd(DirOfFile(aNameIn)+aF2);
             aFin = Virgule(aFin,aT2.in(0));
        }

        if (aVPermut.size() != 0)
           aFin = aFin.permut(aVPermut);

        if (type_im_integral( aTypeOut))
        {
        }
        else
        {
            aFin = Rconv(aFin);
        }

        aFin = reduc_binaire_gen(aFin, aReducX != 0, aReducY != 0, 16, true, 0);
        anOut = Filtre_Out_RedBin_Gen(anOut, aReducX != 0, aReducY != 0);
        aSzROut = aSzOut.mcbyc(aCoefReduc);
        aFin = trans(aFin,aP0);

        if (aKCh!=-1)
           aFin = aFin.kth_proj(aKCh);
        else
        {

            if ((aNbChOut==1) && (aNbChIn==3))
                aFin = (aFin.v0() + aFin.v1() + aFin.v2()) / 3.0;

            if ((aNbChOut==3) && (aNbChIn==1))
               aFin = Virgule(aFin,aFin,aFin);
         }


        if (aVisu)
           anOut = anOut |  Video_Win::WiewAv(aSzROut);

        if (aDyn != 1.0)
           aFin = aFin * aDyn;

        if (type_im_integral(aTypeOut) && (aTypeOut!=GenIm::int4))
        {
            int aVMin,aVMax;
            min_max_type_num(aTypeOut,aVMin,aVMax);
            aFin = Max(aVMin,Min(aVMax-1,aFin));
        }

        ELISE_COPY(rectangle(Pt2di(0,0),aSzROut),aFin,anOut);

        return EXIT_SUCCESS;
    }
    else return EXIT_SUCCESS;
}





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
