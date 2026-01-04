/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include "GateCoincidenceSorterActor.h"
#include "../GateHelpersDict.h"
#include "GateDigiCollectionManager.h"

GateCoincidenceSorterActor::GateCoincidenceSorterActor(py::dict &user_info)
    : GateVDigitizerWithOutputActor(user_info, false) {
  fActions.insert("EndOfEventAction");
  fActions.insert("EndOfRunAction");
}

GateCoincidenceSorterActor::~GateCoincidenceSorterActor() = default;

void GateCoincidenceSorterActor::InitializeUserInfo(py::dict &user_info) {

  GateVDigitizerWithOutputActor::InitializeUserInfo(user_info);
  if (py::len(user_info) > 0 && user_info.contains("window")) {
    fWindowSize = DictGetDouble(user_info, "window"); // nanoseconds
  }
  if (py::len(user_info) > 0 && user_info.contains("offset")) {
    fWindowOffset = DictGetDouble(user_info, "offset"); // nanoseconds
  }
  if (py::len(user_info) > 0 && user_info.contains("multiples_policy")) {
    const auto policy_str = DictGetStr(user_info, "multiples_policy");
    if (policy_str == "RemoveMultiples") {
      fMultiplesPolicy = MultiplesPolicy::RemoveMultiples;
    } else if (policy_str == "TakeAllGoods") {
      fMultiplesPolicy = MultiplesPolicy::TakeAllGoods;
    } else if (policy_str == "TakeWinnerOfGoods") {
      fMultiplesPolicy = MultiplesPolicy::TakeWinnerOfGoods;
    } else if (policy_str == "TakeIfOnlyOneGood") {
      fMultiplesPolicy = MultiplesPolicy::TakeIfOnlyOneGood;
    } else if (policy_str == "TakeWinnerIfIsGood") {
      fMultiplesPolicy = MultiplesPolicy::TakeWinnerIfIsGood;
    } else if (policy_str == "TakeWinnerIfAllAreGoods") {
      fMultiplesPolicy = MultiplesPolicy::TakeWinnerIfAllAreGoods;
    } else {
      Fatal("Unknown multiples policy '" + policy_str + "'");
    }
  }
  if (py::len(user_info) > 0 && user_info.contains("multi_window")) {
    fMultiWindow = DictGetDouble(user_info, "multi_window");
  }
  if (py::len(user_info) > 0 && user_info.contains("min_transaxial_distance")) {
    const auto value = user_info.attr("min_transaxial_distance");
    if (!value.is_none()) {
      fMinTransaxialDistance =
          DictGetDouble(user_info, "min_transaxial_distance");
    }
  }
  if (py::len(user_info) > 0 && user_info.contains("max_axial_distance")) {
    const auto value = user_info.attr("max_axial_distance");
    if (!value.is_none()) {
      fMaxAxialDistance = DictGetDouble(user_info, "max_axial_distance");
    }
  }
  if (py::len(user_info) > 0 && user_info.contains("transaxial_plane")) {
    const auto transaxial_plane_str = DictGetStr(user_info, "transaxial_plane");
    if (transaxial_plane_str == "XY") {
      fTransaxialPlane = TransaxialPlane::XY;
    } else if (transaxial_plane_str == "YZ") {
      fTransaxialPlane = TransaxialPlane::YZ;
    } else if (transaxial_plane_str == "XZ") {
      fTransaxialPlane = TransaxialPlane::XZ;
    } else {
      Fatal("Unknown transaxial plane '" + transaxial_plane_str + "'");
    }
  }
  if (py::len(user_info) > 0 && user_info.contains("sorting_time")) {
    fSortingTime = DictGetDouble(user_info, "sorting_time"); // nanoseconds
  }
  fGroupVolumeDepth = -1;
  fInputDigiCollectionName = DictGetStr(user_info, "input_digi_collection");
}

void GateCoincidenceSorterActor::SetGroupVolumeDepth(const int depth) {
  fGroupVolumeDepth = depth;
}

void GateCoincidenceSorterActor::DigitInitialize(
    const std::vector<std::string> &attributes_not_in_filler) {

  GateVDigitizerWithOutputActor::DigitInitialize({});

  // Set up pointers to track specific attributes
  auto &lr = fThreadLocalVDigitizerData.Get();
  auto &l = fThreadLocalData.Get();

  fTimeSorter.Init(fInputDigiCollection);
  fTimeSorter.OutputIterator().TrackAttribute("GlobalTime", &l.time);
  fTimeSorter.OutputIterator().TrackAttribute("PreStepUniqueVolumeID",
                                              &l.volID);
  fTimeSorter.SetSortingWindow(fSortingTime);
  fTimeSorter.SetMaxSize(fClearEveryNEvents);
}

void GateCoincidenceSorterActor::EndOfEventAction(const G4Event *) {
  auto &l = fThreadLocalData.Get();
  fTimeSorter.Process();
  ProcessTimeSortedSingles();
}

void GateCoincidenceSorterActor::EndOfRunAction(const G4Run *) {
  fTimeSorter.Flush();
  ProcessTimeSortedSingles();

  // TODO finish implementation

  // Make sure everything is output into the root file.
  fOutputDigiCollection->FillToRootIfNeeded(true);
}

void GateCoincidenceSorterActor::ProcessTimeSortedSingles() {
  auto &l = fThreadLocalData.Get();
  auto &iter = fTimeSorter.OutputIterator();
  iter.GoToBegin();
  while (!iter.IsAtEnd()) {

    // TODO implement

    iter++;
  }
  fTimeSorter.MarkOutputAsProcessed();
}
