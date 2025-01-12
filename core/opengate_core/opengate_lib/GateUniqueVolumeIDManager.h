/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#ifndef GateUniqueVolumeIDManager_h
#define GateUniqueVolumeIDManager_h

#include "G4VTouchable.hh"
#include "GateUniqueVolumeID.h"
#include <string>
#include <unordered_map>
#include <utility>

/*
    Global singleton class that manage a correspondence between touchable
    pointer and unique volume ID.
 */

class GateUniqueVolumeIDManager {
public:
  static GateUniqueVolumeIDManager *GetInstance();

  GateUniqueVolumeID::Pointer GetVolumeID(const G4VTouchable *touchable);

  std::vector<GateUniqueVolumeID::Pointer> GetAllVolumeIDs() const;

protected:
  GateUniqueVolumeIDManager();

  static GateUniqueVolumeIDManager *fInstance;

  struct IDArrayTypeHash {
    std::size_t operator()(const GateUniqueVolumeID::IDArrayType &arr) const {
      std::size_t hashValue = 0;
      std::hash<int> intHasher;
      for (const auto &elem : arr) {
        if (elem == -1)
          break;
        hashValue ^=
            intHasher(elem) + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
      }
      return hashValue;
    }
  };

  template <typename T1, typename T2> struct PairHash {
    std::size_t operator()(const std::pair<T1, T2> &p) const {
      std::size_t h1 = std::hash<T1>()(p.first);
      std::size_t h2 = IDArrayTypeHash()(p.second);
      return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
  };

  // Index of name + ID array to VolumeID
  // This map is created on the fly in GetVolumeID
  std::map<std::pair<std::string, GateUniqueVolumeID::IDArrayType>,
           GateUniqueVolumeID::Pointer>
      fToVolumeID;

  /*
    std::unordered_map<std::pair<std::string, GateUniqueVolumeID::IDArrayType>,
                       GateUniqueVolumeID::Pointer,
                       PairHash<std::string, GateUniqueVolumeID::IDArrayType>>
        fToVolumeID;
  */
};

#endif // GateUniqueVolumeIDManager_h
