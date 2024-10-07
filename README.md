# 三角形上の最も近い点

## 概要

このプロジェクトは次の課題を解決するための実装です。

**3点が与えられた定義される平面（三角形）と、別の点が与えられたとき、その点から平面上の最も近い点を見つけて**

### 動機
最近、この問題を説明しようとしましたが、残念ですが、うまく日本語で説明できませんでした。そこで、自分の理解を深め、日本語の線形代数の言葉を学ぶために、これを実装しました。
また、平面上の最も近い点だけでなく、三角形の中にあるかどうかも調べる実装しました。

---

## 説明

1. **平面の定義:**
   - 3つの点 `A`、`B`、`C` があります。
   - ベクトル `BA` と `CA` を作って、外積で平面に垂直な法線ベクトル `N` を求めます。

2. **平面への投影:**
   - 与えられた点 `P` から、点 `A` までのベクトル `PA` を考えますが、そのベクトルは平面に対してまだ垂直じゃないです。
   - `PA` と `N` のドット積を使って投影ベクトルの長さを計算します
   - その長さは法線ベクトルと掛けて、正しい投影ベクトルを得ます。
   - 最終的に、この投影ベクトルを `P` に加えることで、平面上の最も近い点を見つけます。

3. **三角形の中にあるかの確認:**
   - 平面上の最も近い点を見つけた後、その点が三角形の中にあるかどうかを確認します。
   - `AX`、`BX`、`CX` （`X` は投影された点）のベクトルを使って、これらのベクトルの外積を計算します。
   - ドット積を使って結果が全てポジティブの場合、その点は三角形の中にあります。
   - 点が外にある場合、前に説明したほぼ同じように三角形の辺に投影をして、三角形の中の最も近い点を見つけます。

---

## 実装

`getClosestPointOnModel` の関数:
- 平面上の最も近い点を計算します。
- その点が三角形の中にあるか、そうではない場合は最も近い辺の点を見つけます。

```cpp
   //....

   int i_p1 = model.indices[i + 0];
   int i_p2 = model.indices[i + 1];
   int i_p3 = model.indices[i + 2];

   const float* vertData = &model.verticesAndNormals[0];
   glm::vec3 p1(vertData[6 * i_p1], vertData[6 * i_p1 + 1], vertData[6 * i_p1 + 2]);
   glm::vec3 p2(vertData[6 * i_p2], vertData[6 * i_p2 + 1], vertData[6 * i_p2 + 2]);
   glm::vec3 p3(vertData[6 * i_p3], vertData[6 * i_p3 + 1], vertData[6 * i_p3 + 2]);

   // apply model TRS to the points
   p1 = model.TRS * glm::vec4(p1, 0);
   p2 = model.TRS * glm::vec4(p2, 0);
   p3 = model.TRS * glm::vec4(p3, 0);

   glm::vec3 v1 = p2 - p1;
   glm::vec3 v2 = p3 - p1;

   glm::vec3 planeNormal = glm::cross(v1, v2);
   planeNormal = glm::normalize(planeNormal);

   float distanceToPlane = glm::dot(planeNormal, (point - p1));
   distanceToPlane *= -1;

   glm::vec3 translationVector = planeNormal * distanceToPlane;
   glm::vec3 closestPointOnThePlane = point + translationVector;

   // closestPointOnThePlane is the projection on the plane defined by the 3 vertices
   // now make sure the point is inside the triangle
   glm::vec3 _p1 = p1 - closestPointOnThePlane;
   glm::vec3 _p2 = p2 - closestPointOnThePlane;
   glm::vec3 _p3 = p3 - closestPointOnThePlane;

   glm::vec3 u = glm::cross(_p2, _p3);
   glm::vec3 v = glm::cross(_p3, _p1);
   glm::vec3 w = glm::cross(_p1, _p2);

   float uvAngle = glm::dot(u, v);
   float uwAngle = glm::dot(u, w);

   // if any is negative the value is outside the triangle
   // need to do a second projection
   // find the projected point for each and see which one is closest

   if (uvAngle < 0.0f || uwAngle < 0.0f)
   {

      // could extract to a function but keeping it here to vizualize the math better
      glm::vec3 closestPointOnTheEdge1;
      {
         glm::vec3 edge = p2 - p1;
         float t = glm::dot(closestPointOnThePlane - p1, edge) / glm::dot(edge, edge);
         t = glm::clamp(t, 0.0f, 1.0f);
         closestPointOnTheEdge1 = p1 + t * edge;
      }

      glm::vec3 closestPointOnTheEdge2;
      {
         glm::vec3 edge = p3 - p2;
         float t = glm::dot(closestPointOnThePlane - p2, edge) / glm::dot(edge, edge);
         t = glm::clamp(t, 0.0f, 1.0f);
         closestPointOnTheEdge2 = p2 + t * edge;
      }

      glm::vec3 closestPointOnTheEdge3;
      {
         glm::vec3 edge = p1 - p3;
         float t = glm::dot(closestPointOnThePlane - p3, edge) / glm::dot(edge, edge);
         t = glm::clamp(t, 0.0f, 1.0f);
         closestPointOnTheEdge3 = p3 + t * edge;
      }

      // Choose the closest of these points
      float dist1 = glm::length(closestPointOnTheEdge1 - closestPointOnThePlane);
      float dist2 = glm::length(closestPointOnTheEdge2 - closestPointOnThePlane);
      float dist3 = glm::length(closestPointOnTheEdge3 - closestPointOnThePlane);

      if (dist1 < dist2 && dist1 < dist3)
      {
         closestPointOnThePlane = closestPointOnTheEdge1;
      }
      else if (dist2 < dist3)
      {
         closestPointOnThePlane = closestPointOnTheEdge2;
      }
      else
      {
         closestPointOnThePlane = closestPointOnTheEdge3;
      }
   }     

   //....
```

---

## セットアップ

このプロジェクトを動かすためには、以下の依存パッケージをインストールしてください。

```bash
vcpkg install glew glfw3 glm assimp imgui[core,glfw-binding,opengl3-binding] imguizmo
```

---

## 今後の改善

現在の実装は動作しますが、いくつかの改善点があります:

- **コード整理:** より良い整理のためにいくつかのファイルに分かれ。
- **シェーダー管理:** 繰り返しのコードとセットアップを避けるために、シェーダークラスを実装して。
- **カメラの最適化:** カメラ位置をキャッシュして、必要なときだけ再計算。
- **ウィンドウのサイズ変更:** ウィンドウをサイズ変更可能にする。
- **BVHデータ構造:** 高いポリゴンメッシュでのパフォーマンス向上のためにバウンディングボリューム構造（BVH）を実装して。
- **オブジェクトの最適化:** 各フレームでのオブジェクト（点、グリッド、線）の作成と削除を避ける。ただし、現在はパフォーマンスに大きな影響はありません。
- **シェーダーでの計算:** シェーダー内で計算をして、楽しさやパフォーマンス改善の可能性を探る。

---

マヌエル・コレイア・ネヴェス・トマス・ダ・シルバ  
2024年