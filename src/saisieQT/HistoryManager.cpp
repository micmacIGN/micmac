#include "HistoryManager.h"

HistoryManager::HistoryManager():
    _actionIdx(0)
{}

void HistoryManager::push_back(selectInfos &infos)
{
    int sz = _infos.size();

    if (_actionIdx < sz)
    {
        for (int aK=_actionIdx; aK < sz; ++aK)
            _infos.pop_back();
    }

    _infos.push_back(infos);

    _actionIdx++;
}
