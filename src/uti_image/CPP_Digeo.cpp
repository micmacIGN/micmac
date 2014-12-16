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

#include "Digeo/Digeo.h"

// add i_v to the coordinates of the i_nbPoints last points of io_points
void translate_points( list<DigeoPoint> &io_points, size_t i_nbPoints, const Pt2di &i_v )
{
	#ifdef __DEBUG_DIGEO
		if ( i_nbPoints>io_points.size() ){ cerr << "translate_points: trying to translate " << i_nbPoints << " out of " << io_points.size() << endl; exit(EXIT_FAILURE); }
	#endif
	const double tx = (double)i_v.x,
	             ty = (double)i_v.y;
	list<DigeoPoint>::reverse_iterator itPoint = io_points.rbegin();
	while ( i_nbPoints-- ){
		itPoint->x = itPoint->x+tx;
		itPoint->y = itPoint->y+ty;
		itPoint++;
	}
}

int Digeo_main( int argc, char **argv )
{
	ElTimer chrono;

	if ( argc!=4 )
	{
		cerr << "Digeo: usage : mm3d Digeo input_filename -o output_filename" << endl;
		return EXIT_FAILURE;
	}

	std::string inputName  = argv[1];
	std::string outputName = argv[3];

	cAppliDigeo appli;
	appli.times()->start();
	appli.loadImage( inputName );
	appli.times()->stop("pyramid structure");

	cImDigeo &image = appli.getImage();

    if ( appli.isVerbose() )
    {
       cout << "number of tiles : " << appli.NbInterv() << endl;
       cout << "tile size : " << appli.Params().DigeoDecoupageCarac().Val().SzDalle() << endl;
       cout << "margin : " << appli.Params().DigeoDecoupageCarac().Val().Bord() << endl;
    }

    list<DigeoPoint> total_list;
    for (int aKBox = 0 ; aKBox<appli.NbInterv() ; aKBox++)
    {
        appli.LoadOneInterv(aKBox);  // Calcul et memorise la pyramide gaussienne

        if ( !appli.doComputeCarac() ) continue;

        Box2di box = appli.getInterv( aKBox );
        if ( appli.isVerbose() ) cout << "processing tile " << aKBox << " of origin " << box._p0 << " and size " << box.sz() << endl;

        image.detect();

        if ( appli.isVerbose() ) cout << "\t" << image.getNbFeaturePoints() << " feature points" << endl;

        image.orientateAndDescribe();

        if ( appli.doPlotPoints() ) image.plotPoints();

        size_t nbTilePoints = image.addAllPoints( total_list );
        translate_points( total_list, nbTilePoints, box._p0 );

        if ( appli.isVerbose() ) cout << "\t" << nbTilePoints << " points" << endl;
    }

	appli.mergeOutputs();

	if ( appli.doGenerateConvolutionCode() )
	{
		appli.generate_convolution_code<U_INT2>();
		appli.generate_convolution_code<REAL4>();
	}
	else if ( appli.isVerbose() && ( appli.nbSlowConvolutionsUsed<U_INT2>()!=0 || appli.nbSlowConvolutionsUsed<REAL4>()!=0 ) )
		cout << "skipping convolution code generation" << endl;

	ElTimer chronoSave;
	cout << total_list.size() << " points" << endl;
	if ( !DigeoPoint::writeDigeoFile( outputName, total_list ) ) cerr << "Digeo: ERROR: unable to save points to file " << outputName << endl;
	double saveTime = chronoSave.uval();

	if ( appli.doShowTimes() )
	{
		MapTimes *times = (MapTimes*)appli.times();
		double totalMonitoredTime = saveTime+times->totalTime();

		cout << "Total time = " << chrono.uval() << " (" << totalMonitoredTime << ")" << endl;
		times->printTimes( "\t" );
		cout << "\tsave :" << saveTime << endl;
	}

	cout << "nb computed gradient = " << appli.nbComputedGradients() << endl;

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
