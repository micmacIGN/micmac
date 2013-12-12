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

cElPath strToPath( const string i_str )
{
   cElPath p( i_str );
   cout << "[" << i_str << "] " << endl;
   p.trace();
   cout << "--unix-----> " << p.str_unix() << endl;
   cout << "--windows--> " << p.str_windows() << endl;
   cout << endl;
   return p;
}

void bench_path()
{
   // test la classe cElPath

   cout << "current directory = ";
   getCurrentDirectory().trace();

   cout << "null directory (invalid) = ";
   cElPath().trace();

   cElPath p0 = strToPath( "titi/toto" ),
	   p1 = strToPath( "titi\\toto" ),
	   p2 = strToPath( "titi/./toto/../toto/" ),
	   p3 = strToPath( "titi/toto/../../../tata" ),
	   p4 = strToPath( "c:\\program files\\micmac\\bin" );
   
   bench_error( p0!=p1, "p0!=p1" );
   bench_error( p1!=p2, "p1!=p2" );
}

void bench_command()
{
   // test la classe cElCommand et ses classes connexes
   bench_path();
   string command = "\"d:/dev/culture3d/bin/mm3d\" Apero \"d:/dev/culture3d/include/XML_MicMac/Apero-Glob.xml\"  DirectoryChantier=./  \"+PatternAllIm=.*.JPG\"  +AeroOut=-Arbitrary +Ext=dat +ModeleCam=eCalibAutomPhgrStdBasic +FileLibereParam=Param-Fraser.xml DoCompensation=1 +SeuilFE=-1.000000 +TetaLVM=0.010000 +CentreLVM=0.100000 +RayFEInit=0.850000 +CalibIn=-#@LL?~~XXXXXXXXXX +AeroIn=-#@LL?~~XXXXXXXXXX +VitesseInit=4 +PropDiagU=1.000000 +ValDec=eLiberte_Phgr_Std_Dec +ValDecPP=eLiberte_Dec1 +ValAffPP=eLiberteParamDeg_1 +ValAff=eLiberte_Phgr_Std_Aff";
   cout << "[" << command << "]" << endl;
   ::System( command );
}
