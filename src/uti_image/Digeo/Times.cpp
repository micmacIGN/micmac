#include "Times.h"

#include <iomanip>

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

bool MapTimes::Record::hasRecord( const std::string &aName ) const
{
	list<ItRecord>::const_iterator itChild = mSubRecords.begin();
	while ( itChild!=mSubRecords.end() )
	{
		if ( (**itChild).mName==aName || (**itChild).hasRecord(aName) ) return true;
		itChild++;
	}
	return false;
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

double MapTimes::stop( const char *aName )
{
	const string name(aName);
	if ( mCurrent==mRecords.begin() )
	#ifdef __DEBUG_TIMES
		ELISE_DEBUG_ERROR( true, "MapTimes::stop", "mCurrent==mRecords.begin() (stopping a Record never started)");
		ELISE_DEBUG_ERROR( mCurrent->hasRecord(name), "MapTimes::stop", "record of name [" << name << "] has a descendant with the same name" );
	#else
		return 0.;
	#endif

	double t = mCurrent->mTimer.uval();

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
		mRecords.erase(mCurrent);
		mCurrent = newCurrent;
	}

	return t;
}

void MapTimes::printTimes( const string &i_prefix, ostream &io_ostream ) const
{
	if ( mRecords.begin()!=mRecords.end() ) mRecords.front().printTimes( i_prefix, io_ostream );
}

double MapTimes::getRecordTime( const std::string &aName ) const
{
	list<Record>::const_iterator itRecord = mRecords.begin();
	while ( itRecord!=mRecords.end() )
	{
		if ( itRecord->mName==aName ) return itRecord->mTime;
		itRecord++;
	}
	return 0.;
}

#ifdef ELISE_unix
	double getTime(){ return ElTimeOfDay(); }
#else
	#include <sys/time.h>
	double getTime()
	{
		struct timeval tv;
		gettimeofday( &tv, NULL ); 
		return tv.tv_sec+( tv.tv_usec/1e6 );
	}
#endif
