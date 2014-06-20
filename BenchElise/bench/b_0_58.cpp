/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/

#ifndef __TRACE_SYSTEM__
   #define __TRACE_SYSTEM__
#endif

#include "StdAfx.h"
#include "general/cElCommand.h"
#include "bench.h"

void bench_error( bool i_testHasFailed, const string &i_msg )
{
   if ( i_testHasFailed )
   {
      cerr << i_msg << endl;
      exit( EXIT_FAILURE );
   }
}

ctPath strToPath( const string i_str )
{
   ctPath p( i_str );
   cout << i_str << " -> [" << p.str() << "] " << endl;
   //p.trace();
   return p;
}

void bench_path()
{
   // test la classe ctPath

   cout << "current directory = ";
   getWorkingDirectory().trace();

   cout << "--- test path normalized form" << endl;
   ctPath p0 = strToPath( "titi/to to/" ),
	  p1 = strToPath( "titi\\to to" ),
	  p2 = strToPath( "titi/./toto/../to to/" );
   bench_error( p0!=p1, "p0!=p1" );
   bench_error( p1!=p2, "p1!=p2" );
   cout << "ok\n" << endl;

   cout << "--- test cElFilename creation/deletion" << endl;
   cElFilename filename( "balo_file" );
   bench_error( filename.exists(), filename.str_unix()+" already exists" );
   bench_error( !filename.create(), string("cannot create file [")+filename.str_unix()+"]" );
   #if (!ELISE_windows)
		mode_t wantedRights = 0755, rights;
		bench_error( !filename.setRights( wantedRights ), string("cannot set rights on [")+filename.str_unix()+"]" );
		bench_error( !filename.getRights( rights ), string("cannot get rights on [")+filename.str_unix()+"]" );
		bench_error( wantedRights!=rights, string("a error occured while setting or getting rights on [")+filename.str_unix()+"]" );
	#endif
   bench_error( !filename.remove(), string("cannot remove file [")+filename.str_unix()+"]" );
   
   cout << "--- test ctPath creation/deletion" << endl;
   ctPath sampleDir( "balo_path" );
   bench_error( !sampleDir.create(), string("cannot create directory [")+sampleDir.str()+"]" );
   bench_error( !sampleDir.removeEmpty(), string("cannot remove directory [")+sampleDir.str()+"]" );
   cout << "ok\n" << endl;
   
	cout << "--- creating/deleting a basic tree" << endl;
	ctPath treeName("toto");
	bench_error( treeName.exists(), string("tree [")+treeName.str()+"] already exists" );
	bench_error( !ctPath("toto").create() ||
	             !ctPath("toto/a").create() ||
	             !ctPath("toto/b").create() ||
	             !ctPath("toto/c").create() ||
	             !cElFilename("toto/a/aa").create() ||
	             !cElFilename("toto/a/aabbb").create() ||
	             !cElFilename("toto/a/aabbbbxxx").create() ||
	             !cElFilename("toto/c/cc").create() ||
	             !cElFilename("toto/c/ccddd").create() ||
	             !cElFilename("toto/c/balo").create(),
	             "unable to create a basic tree" );
	bench_error ( !ctPath("toto").isAncestorOf( ctPath("toto/a/") ), "directory \"toto\" is not an ancestor of \"toto/a\", which is odd" );
	bench_error ( !ctPath("toto").isAncestorOf( cElFilename("toto/c/balo") ), "directory \"toto\" is not an ancestor of \"toto/c/balo\", which is odd" );
	bench_error ( ctPath("toto").isAncestorOf( cElFilename("../toto") ), "directory \"toto\" is an ancestor of \"../titi\", which is odd" );
	bench_error( !ctPath("toto").remove(), "failed to delete a basic tree" );
	cout << "ok" << endl;
}

void bench_command()
{
   // test la classe cElCommand et ses classes connexes
   bench_path();
   //string command = "\"d:/dev/culture3d/bin/mm3d\" Apero \"d:/dev/culture3d/include/XML_MicMac/Apero-Glob.xml\"  DirectoryChantier=./  \"+PatternAllIm=.*.JPG\"  +AeroOut=-Arbitrary +Ext=dat +ModeleCam=eCalibAutomPhgrStdBasic +FileLibereParam=Param-Fraser.xml DoCompensation=1 +SeuilFE=-1.000000 +TetaLVM=0.010000 +CentreLVM=0.100000 +RayFEInit=0.850000 +CalibIn=-#@LL?~~XXXXXXXXXX +AeroIn=-#@LL?~~XXXXXXXXXX +VitesseInit=4 +PropDiagU=1.000000 +ValDec=eLiberte_Phgr_Std_Dec +ValDecPP=eLiberte_Dec1 +ValAffPP=eLiberteParamDeg_1 +ValAff=eLiberte_Phgr_Std_Aff";
   //cout << "[" << command << "]" << endl;
   //::System( command );
}
