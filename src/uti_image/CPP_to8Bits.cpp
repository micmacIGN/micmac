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

#define DEF_OFSET -13976


int to8Bits_main(int argc,char ** argv)
{
    std::string aNameIn;
    std::string aNameOut;
    REAL EcMin = 0.0;


    INT Brd = -1;
    INT NbIter =-1;
    REAL Dyn = 1.0;

    double Offset = DEF_OFSET;
    double NewOffset = DEF_OFSET;
    INT Circ = 0;
    INT Coul  = -1;
    INT AdaptMinMax = 0;
    INT AdaptMin    = 0;
    INT EqHisto = 0;
    double aStepH = 1.0;

    INT IS1 =-(1<<15);
    REAL GS1 = 0;
    INT WS1 = 0;
    std::string aNameType = "";


    INT aStrip = -1;
    L_Arg_Opt_Tiff aLArgTiff = Tiff_Im::Empty_ARG;
    std::string aNameCompr ="";

    INT aTestVals=0;
    REAL aStep = -1;
    INT aVisuAff = 0;

    int aCanTileFile = 0;

    Pt2dr aP0Crop(0.0,0.0);
    Pt2dr aP1Crop(1.0,1.0);
    bool  aCropIsAbs = false;

    double Big      = 1e30;
    double ForceMax = -2*Big;
    double ForceMin =  2*Big;
    std::string Mask="";
    std::string BoucheMask="";

    bool UseSigne = true;
    bool ToXV = false;



    ElInitArgMain
    (
    argc,argv,
                LArgMain()  << EAMC(aNameIn, "Image", eSAM_IsExistFile),
    LArgMain()  << EAM(EcMin,"EcMin",true)
                    << EAM(aNameOut,"Out",true)
                    << EAM(Brd,"Brd",true)
                    << EAM(NbIter,"NbIter",true)
                    << EAM(Dyn,"Dyn",true)
                    << EAM(Offset,"Offset",true)
                    << EAM(NewOffset,"NewOffset",true)
                    << EAM(Circ,"Circ",true)
                    << EAM(Coul,"Coul",true)
                    << EAM(AdaptMinMax,"AdaptMinMax",true)
                    << EAM(AdaptMin,"AdaptMin",true)
                    << EAM(IS1,"IS1",true)
                    << EAM(GS1,"GS1",true)
                    << EAM(WS1,"WS1",true)
                    << EAM(aNameType,"Type",true,"Type", eSAM_None, ListOfVal(GenIm::bits1_msbf, ""))
                    << EAM(aStrip,"Strip",true)
                    << EAM(aNameCompr,"Compr",true)
                    << EAM(aTestVals,"TestVals",true)
                    << EAM(aStep,"Step",true)
                    << EAM(aVisuAff,"VisuAff",true)
                    << EAM(aP0Crop,"P0Crop",true,"!!!  crop in rel (between 0 & 1)")
                    << EAM(aP1Crop,"P1Crop",true,"!!!  crop in rel (between 0 & 1)")
                    << EAM(aCropIsAbs,"CropIsAbs",true," Set to true if crop is not rel ")
                    << EAM(aCanTileFile,"CanTileFile",true)
                    << EAM(EqHisto,"EqHisto",true)
                    << EAM(aStepH,"StepH",true)
                    << EAM(ForceMax,"ForceMax",true)
                    << EAM(ForceMin,"ForceMin",true)
                    << EAM(Mask,"Mask",true)
                    << EAM(BoucheMask,"BoucheMask",true)
                    << EAM(UseSigne,"UseSigne",true)
                    << EAM(ToXV,"2XV",true)
    );

    if(!MMVisualMode)
    {
        if ((ForceMax> -Big) || (ForceMin < Big))
            AdaptMinMax = true;


        Tiff_Im::COMPR_TYPE aModeCompr = Tiff_Im::No_Compr;
        if (aNameCompr != "")
            aModeCompr = Tiff_Im::mode_compr(aNameCompr);

        if (! aCanTileFile)
            aLArgTiff =  aLArgTiff+ Arg_Tiff(Tiff_Im::AFileTiling(Pt2di(-1,-1)));

        if (aStrip ==0)
            aLArgTiff = aLArgTiff +  Arg_Tiff(Tiff_Im::ANoStrip());
        else if (aStrip >0)
            aLArgTiff = aLArgTiff +  Arg_Tiff(Tiff_Im::AStrip(aStrip));


        GenIm::type_el aTypeOut = GenIm::u_int1;
        if (aNameType!="")
            aTypeOut = type_im(aNameType);

        if (Coul < 0)
            Coul = Circ;

        // Tiff_Im tiff = Tiff_Im::StdConvGen(aNameIn.c_str(),1,true,false);
        // MPD , sinon RGB => Black
        Tiff_Im tiff = Tiff_Im::StdConvGen(aNameIn.c_str(),-1,true,false);



        GenIm::type_el aType = tiff.type_el();

        cout << "Types = "
             << (INT) aType <<  " "
             << (INT) GenIm::int2 <<  " "
             << (INT) GenIm::u_int2 << "\n";

        bool DefOffset= false;
        if (Offset == DEF_OFSET)
        {
            DefOffset= true;
            if (
                    (nbb_type_num(aTypeOut) == nbb_type_num(aType))
                    && (type_im_integral(aTypeOut) && type_im_integral(aType))
                    )
            {
                INT v_minOut,v_maxOut;
                INT v_minIn,v_maxIn;
                min_max_type_num(aTypeOut,v_minOut,v_maxOut);
                min_max_type_num(aType,v_minIn,v_maxIn);
                Offset = ((v_maxOut-v_maxIn) + (v_minOut-v_minIn))/2;
            }
            else
                Offset = 0;
        }


        if (aNameOut == "")
        {
            std::string aPost = "_8Bits.tif";
            if (aNameType !="")
                aPost = aNameType +".tif";
            if (IsPostfixed(aNameIn))
                aNameOut = StdPrefix(aNameIn)+std::string(aPost);
            else
                aNameOut = aNameIn+std::string(aPost);
        }

        Disc_Pal aP1 =  Disc_Pal::PCirc(256);
        Elise_colour * Cols = aP1.create_tab_c();
        Cols[0] = Elise_colour::gray(GS1);
        Disc_Pal aP2 (Cols,256);



        Pt2di aP0_Out  = round_ni(aP0Crop.mcbyc(Pt2dr(tiff.sz())));
        Pt2di aP1_Out  = round_ni(aP1Crop.mcbyc(Pt2dr(tiff.sz())));

        if ( (EAMIsInit(&aP0Crop) || EAMIsInit(&aP1Crop)) && (!aCropIsAbs))
        {
            for (int aK=0 ; aK< 10 ; aK++)
            {
                std::cout << "Warnn you use crop with rel, Sz=" << (aP1_Out-aP0_Out) << "\n";
            }
        }
        if (aCropIsAbs)
        {
            aP0_Out = round_ni(aP0Crop);
            aP1_Out = round_ni(aP1Crop);
        }

        Tiff_Im TiffOut  =
                Coul                ?
                    Tiff_Im
                    (
                        aNameOut.c_str(),
                        (aP1_Out-aP0_Out),
                        aTypeOut,
                        aModeCompr,
                        aP2, // Disc_Pal::PCirc(256)
                        aLArgTiff
                        )                    :
                    Tiff_Im
                    (
                        aNameOut.c_str(),
                        (aP1_Out-aP0_Out),
                        aTypeOut,
                        aModeCompr,
                        // Tiff_Im::BlackIsZero,
                        tiff.phot_interp(),
                        aLArgTiff
                        );

        Symb_FNum  FoncInit(Rconv(tiff.in(0)));
        Fonc_Num  fRes = 0.0;


        if ((Brd > 0) && (NbIter >0))
        {
            Fonc_Num Masq =  (FoncInit!= -(1<<15)) ;

            Symb_FNum  Fonc (FoncInit * Masq);
            Symb_FNum  Pond (tiff.inside()*Masq);

            Fonc_Num fSom = Virgule(Rconv(Pond),Fonc,ElSquare(Fonc));
            for (INT k=0; k< NbIter ; k++)
                fSom = rect_som(fSom,Brd)/ElSquare(1.0+2.0*Brd);  // Pour Eviter les divergences
            Symb_FNum  S012 (fSom);

            Symb_FNum s0 (Rconv(S012.v0()));
            Symb_FNum s1 (S012.v1()/s0);
            Symb_FNum s2 (S012.v2()/s0-Square(s1) + EcMin);
            Symb_FNum ect  (sqrt(Max(0.01,s2)));
            fRes = 255*erfcc((tiff.in()-s1)/ect);
        }
        else
        {
            if (NewOffset!= DEF_OFSET)
            {
                cout << "Dyn= " << Dyn  << " NEW OFF  " << NewOffset << "\n";
                fRes =   (FoncInit + NewOffset) *Dyn;
                if (UseSigne && signed_type_num(aType))
                {
                    fRes = fRes+128;
                }
            }
            else
            {
                cout << "Dyn= " << Dyn  << " OFF  " << Offset << "\n";
                fRes = FoncInit ;
                if (Offset != 0)
                    fRes = fRes + Offset;
                fRes = fRes * Dyn;
                std::cout << "SIGNED " << signed_type_num(aType) << " DO " << DefOffset << "\n";
                if ( (Offset==0) &&  DefOffset && (UseSigne&&signed_type_num(aType)))
                {
                    fRes = fRes+128;
                }
            }
        }

        if (aStep >0)
            fRes = aStep * round_ni(fRes/aStep);


        Im1D_REAL8 aHist(1);
        if (AdaptMinMax || AdaptMin || EqHisto)
        {
            REAL GMin,GMax;
            ELISE_COPY(tiff.all_pts(),Rconv(tiff.in()),VMax(GMax)|VMin(GMin));
            if (ForceMax > -Big)
                GMax = ForceMax;
            if (ForceMin < Big)
                GMin = ForceMin;

            cout << "MIN MAX = " << GMin << " " << GMax << "\n";
            if (AdaptMinMax)
                fRes = (tiff.in()-GMin) * ((Dyn*255.0)  / ElMax(GMax-GMin,1e-2));
            else if (AdaptMin)
                fRes = Min((tiff.in()-GMin),255);
            else if (EqHisto)
            {
                aHist = Im1D_REAL8(2+round_ni((GMax-GMin)/aStepH),0.0);
                ELISE_COPY
                        (
                            tiff.all_pts().chc(round_ni((tiff.in()-GMin)/aStepH)),
                            1,
                            aHist.histo()
                            );
                double * aDH = aHist.data();
                int aNbH = aHist.tx();
                for (int anX = 1; anX<aNbH ; anX++)
                    aDH[anX] +=  aDH[anX-1];
                ELISE_COPY
                        (
                            aHist.all_pts(),
                            aHist.in() * 255.0 / aDH[aNbH-1],
                        aHist.out()
                        );
                fRes = aHist.in()[round_ni((tiff.in()-GMin)/aStepH)];
            }
            //        fRes = Max(0,Min(255,round_ni(fRes)));
        }

        if (Circ)
        {
            fRes = Max(1,mod(round_ni(fRes),256));
        }
        else
        {
            if (type_im_integral(aTypeOut) && (aTypeOut!= GenIm::int4))
            {
                INT  v_min,v_max;
                min_max_type_num(aTypeOut,v_min,v_max);
                cout << "MAX MIN " << v_min << " " << v_max << "\n";
                fRes = Min(v_max-1,Max(v_min,round_ni(fRes)));
            }
        }






        if (Mask !="")
        {
            Tiff_Im tMasq = Tiff_Im::StdConvGen(Mask.c_str(),1,true,false);
            Fonc_Num fBM = 128;
            if (BoucheMask!="")
                fBM =  Tiff_Im::StdConv(BoucheMask.c_str()).in(0);
            fRes = fRes * tMasq.in(0) + fBM * (!tMasq.in(0));
        }

        if (WS1)
        {
            fRes =  (FoncInit!=IS1)*fRes;
        }

        if (aTestVals)
        {
            cout << "READ : \n";
            while(1)
            {
                INT x,y,v,v2;
                cin >> x >> y;
                ELISE_COPY
                        (
                            rectangle(Pt2di(x,y),Pt2di(x+1,y+1)),
                            Virgule(tiff.in(),fRes),
                            Virgule(sigma(v),sigma(v2))
                            );
                cout << "V = " << v <<  " V2 = " << v2 << "\n";
            }
        }

        ELISE_COPY
                (
                    TiffOut.all_pts(),
                    trans(fRes,aP0_Out),
                    TiffOut.out() | (aVisuAff ? Video_Win::WiewAv(tiff.sz()) : Output::onul())
                    );



        if (ToXV)
        {
            /*
        std::string aCom =  std::string("convert -quality 100 ")
                           + aNameOut +  std::string(" ")
                           + StdPrefix(aNameOut) + std::string(".jpg ");
*/
            std::string aDir,aNewName;
            SplitDirAndFile(aDir,aNewName,aNameOut);
            string outputDirectory = ( isUsingSeparateDirectories()?MMTemporaryDirectory():aDir+"Tmp-MM-Dir/" );
            ELISE_fp::MkDir(outputDirectory);

            std::string aCom =  g_externalToolHandler.get( "convert" ).callName() + " "
                    + aNameOut +  std::string(" ")
                    + outputDirectory+StdPrefix(aNewName) + std::string(".gif ");

            system_call(aCom.c_str());
            ELISE_fp::RmFile( aNameOut );
        }

        // convert -quality 100 Z_Num3_DeZoom8_LeChantier_8Bits.tif  Z_Num3_DeZoom8_LeChantier_8Bits.jpg

    }
    return EXIT_SUCCESS;

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
