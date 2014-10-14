#include "StdAfx.h"
#include "Times.h"

using namespace std;

double MapTimes::stop( const char *i_name )
{
	double t = mTimer.uval();
	const string name(i_name);
	map<string,double>::iterator it = mRecords.find(name);
	if ( it==mRecords.end() )
		mRecords.insert( pair<string,double>(name,t) );
	else
		it->second += t;
	return t;
}

double MapTimes::get( const string &i_name ) const
{
	map<string,double>::const_iterator it = mRecords.find(i_name);
	if ( it==mRecords.end() ) return 0.;
	return it->second;
}

double MapTimes::totalTime() const
{
	double res = 0.;
	map<string,double>::const_iterator it = mRecords.begin();
	while ( it!=mRecords.end() )
		res += (*it++).second;
	return res;
}

void MapTimes::printTimes( const string &i_prefix, ostream &io_ostream ) const
{
	// get name length
	size_t nameMaxLength = 0;
	map<string,double>::const_iterator it = mRecords.begin();
	while ( it!=mRecords.end() )
	{
		if ( it->first.length()>nameMaxLength ) nameMaxLength = it->first.length();
		it++;
	}

	double scale = 100./totalTime();
	it = mRecords.begin();
	while ( it!=mRecords.end() )
	{
		cout << i_prefix << it->first << string( nameMaxLength-it->first.length(), ' ' ) << " : " << it->second << " (" << setw(5) << it->second*scale << "%)" << endl;
		it++;
	}
}
