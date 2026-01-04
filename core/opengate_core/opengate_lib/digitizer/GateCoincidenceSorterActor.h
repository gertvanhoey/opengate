/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#ifndef GateCoincidenceSorterActor_h
#define GateCoincidenceSorterActor_h

#include "GateTimeSorter.h"
#include "GateVDigitizerWithOutputActor.h"
#include <G4Cache.hh>
#include <pybind11/stl.h>

namespace py = pybind11;

class GateCoincidenceSorterActor : public GateVDigitizerWithOutputActor {

public:
  explicit GateCoincidenceSorterActor(py::dict &user_info);

  ~GateCoincidenceSorterActor() override;

  void InitializeUserInfo(py::dict &user_info) override;

  void StartSimulationAction() override;

  void EndOfEventAction(const G4Event *event) override;

  void EndOfRunAction(const G4Run *run) override;

  void SetGroupVolumeDepth(int depth);

protected:
  void DigitInitialize(
      const std::vector<std::string> &attributes_not_in_filler) override;

  enum class MultiplesPolicy {
    RemoveMultiples,
    TakeAllGoods,
    TakeWinnerOfGoods,
    TakeIfOnlyOneGood,
    TakeWinnerIfIsGood,
    TakeWinnerIfAllAreGoods
  };
  enum class TransaxialPlane { XY, YZ, XZ };

  // Coincidence sorter parameters.
  double fWindowSize;
  double fWindowOffset;
  MultiplesPolicy fMultiplesPolicy;
  bool fMultiWindow;
  std::optional<double> fMinTransaxialDistance{};
  std::optional<double> fMaxAxialDistance{};
  TransaxialPlane fTransaxialPlane{TransaxialPlane::XY};
  int fGroupVolumeDepth;
  double fSortingTime;

  struct TemporaryStorage {
    TemporaryStorage(GateDigiCollection *input, GateDigiCollection *output,
                     const std::string &name_suffix);

    GateDigiCollection *digis;
    std::unique_ptr<GateDigiAttributesFiller> fillerIn;
    std::unique_ptr<GateDigiAttributesFiller> fillerOut;
  };

  GateTimeSorter fTimeSorter;

  std::unique_ptr<TemporaryStorage> fCurrentStorage;
  std::unique_ptr<TemporaryStorage> fFutureStorage;

  void ProcessTimeSortedSingles();

  struct threadLocalT {
    GateUniqueVolumeID::Pointer *volID;
    double *time;
    G4ThreeVector *pos;
  };

  G4Cache<threadLocalT> fThreadLocalData;
};

#endif // GateCoincidenceSorterActor_h
