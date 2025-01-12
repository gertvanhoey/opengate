/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include "GateUniqueVolumeIDManager.h"
#include "GateHelpers.h"
#include <shared_mutex>

G4Mutex GetVolumeIDMutex = G4MUTEX_INITIALIZER;
std::shared_mutex rw_lock;

GateUniqueVolumeIDManager *GateUniqueVolumeIDManager::fInstance = nullptr;

GateUniqueVolumeIDManager *GateUniqueVolumeIDManager::GetInstance() {
  if (fInstance == nullptr)
    fInstance = new GateUniqueVolumeIDManager();
  return fInstance;
}

GateUniqueVolumeIDManager::GateUniqueVolumeIDManager() = default;

GateUniqueVolumeID::Pointer
GateUniqueVolumeIDManager::GetVolumeID(const G4VTouchable *touchable) {
  // This function is potentially called a large number of time (every hit)
  // It worth it to make it faster if possible (unsure how).
  // However, without the mutex, it sef fault sometimes in MT mode.
  // Maybe due to some race condition around the shared_ptr. I don't know.
  // With the mutex, no seg fault.

  constexpr bool useReadWriteLock = false;

  if (!useReadWriteLock) {

    G4AutoLock mutex(&GetVolumeIDMutex);

    // https://geant4-forum.web.cern.ch/t/identification-of-unique-physical-volumes-with-ids/2568/3
    // ID
    auto name = touchable->GetVolume()->GetName();
    auto id = GateUniqueVolumeID::ComputeArrayID(touchable);
    // Search if this touchable has already been associated with a unique volume
    // ID
    if (fToVolumeID.count({name, id}) == 0) {
      // It does not exist, so we create it.
      auto uid = GateUniqueVolumeID::New(touchable);
      fToVolumeID[{name, id}] = uid;
    }
    return fToVolumeID.at({name, id});

  } else {

    auto name = touchable->GetVolume()->GetName();
    auto id = GateUniqueVolumeID::ComputeArrayID(touchable);

    // Lock the read access lock before checking if the touchable has already
    // been associated with a unique volume ID.
    std::shared_lock<std::shared_mutex> readLock(rw_lock);
    auto it = fToVolumeID.find({name, id});
    if (it != fToVolumeID.end()) {
      return it->second;
    } else {
      // It does not exist yet, so we will create it. But first we need to
      // obtain exclusive write access.
      readLock.unlock();
      std::unique_lock<std::shared_mutex> writeLock(rw_lock);
      // There is a chance that another thread has created the unique ID in the
      // time interval between unlocking the read lock and locking the write
      // lock, so we have to check again if the unique volume ID exists already.
      auto it2 = fToVolumeID.find({name, id});
      if (it2 != fToVolumeID.end()) {
        // Return the existing ID.
        return fToVolumeID.at({name, id});
      } else {
        // Create and return a new volume ID.
        auto uid = GateUniqueVolumeID::New(touchable);
        fToVolumeID[{name, id}] = uid;
        return uid;
      }
    }
  }
}

std::vector<GateUniqueVolumeID::Pointer>
GateUniqueVolumeIDManager::GetAllVolumeIDs() const {
  std::vector<GateUniqueVolumeID::Pointer> l;
  for (const auto &x : fToVolumeID) {
    l.push_back(x.second);
  }
  return l; // copy
}
