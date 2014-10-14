#include "StdAfx.h"
#include "Times.h"

using namespace std;

//----------------------------------------------------------------------
// methods of class MapTimes::Record
//----------------------------------------------------------------------

double MapTimes::Record::totalTime() const
{
	if ( &(*mFather)!=this ) return mTime;

	double res = 0.;
	for ( list<ItRecord>::const_iterator it = mSubRecords.begin(); it!=mSubRecords.end(); it++ )
		res += ( **it ).mTime;
	return res;
}

void MapTimes::Record::printTimes( const string &i_prefix, ostream &io_ostream ) const
{
	// get name length
	size_t nameMaxLength = 0;
	for ( list<ItRecord>::const_iterator it = mSubRecords.begin(); it!=mSubRecords.end(); it++ )
		if ( (**it).mName.length()>nameMaxLength ) nameMaxLength = (**it).mName.length();

	double scale = 100./totalTime();
	for ( list<ItRecord>::const_iterator it = mSubRecords.begin(); it!=mSubRecords.end(); it++ )
	{
		const Record &record = **it;
		cout << i_prefix << record.mName << string( nameMaxLength-record.mName.length(), ' ' ) << " : " << record.mTime << " (" << setw(5) << record.mTime*scale << "%)" << endl;
	}
}


//----------------------------------------------------------------------
// methods of class MapTimes
//----------------------------------------------------------------------

MapTimes::MapTimes()
{
	// create the root record
	mRecords.push_back( Record() );
	mCurrent = mRecords.begin();
	mCurrent->mFather = mCurrent;
}

void MapTimes::start()
{
	mRecords.push_back( Record() );
	ItRecord current = mRecords.end();
	current--;
	current->mFather = mCurrent;
	mCurrent = current;
}

// return i_list.end() if a record of that name does not exist
list<MapTimes::ItRecord>::iterator MapTimes::getRecord( const string &i_name, list<ItRecord> &i_list )
{
	for ( list<ItRecord>::iterator it=i_list.begin(); it!=i_list.end(); it++ )
		if ( (**it).mName==i_name ) return it;
	return i_list.end();
}

double MapTimes::stop( const char *i_name )
{
	if ( mCurrent==mRecords.begin() )
	#ifdef __DEBUG_TIMES
		__elise_debug_error("MapTimes::stop: mCurrent==mRecords.begin() (stopping a Record never started)");
	#else
		return;
	#endif

	double t = mCurrent->mTimer.uval();

	const string name(i_name);
	list<ItRecord>::iterator record = getRecord( name, mCurrent->mFather->mSubRecords );

	if ( record==mCurrent->mFather->mSubRecords.end() )
	{
		// no record exists with this name at the current level
		mCurrent->mName = name;
		mCurrent->mTime = t;
		mCurrent->mFather->mSubRecords.push_back(mCurrent);
		mCurrent = mCurrent->mFather;
	}
	else
	{
		// there is already a record with this name at this level
		(**record).mTime += t;
		ItRecord newCurrent = mCurrent->mFather;
		mRecords.erase( mCurrent );
		mCurrent = newCurrent;
	}

	return t;
}

void MapTimes::printTimes( const string &i_prefix, ostream &io_ostream ) const
{
	list<Record>::const_iterator it = mRecords.begin();
	if ( it!=mRecords.end() ) (*it++).printTimes( i_prefix, io_ostream );
	while ( it!=mRecords.end() )
	{
		if ( it->mSubRecords.begin()!=it->mSubRecords.end() )
		{
			io_ostream << endl;
			io_ostream << i_prefix << it->mName << endl;
			it->printTimes( i_prefix+'\t', io_ostream );
		}
		it++;
	}
}
