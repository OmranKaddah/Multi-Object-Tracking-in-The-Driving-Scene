/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep (osep -at- vision.rwth-aachen.de)

rwth_mot framework is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

rwth_mot framework is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
rwth_mot framework; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "utils_bounding_box.h"
//std
#include<cmath>
// eigen
#include <Eigen/Core>
// openCV
#include "opencv2/imgproc.hpp"

//qhull
#include"libqhullcpp/Qhull.h"
#include"libqhullcpp/RboxPoints.h"

// utils
#include "utils_filtering.h"
#include "camera.h"
#include "utils_common.h"


namespace SUN {
    namespace utils {
        namespace bbox {

            Eigen::Vector4d Intersection2d(const Eigen::Vector4d &rect1, const Eigen::Vector4d &rect2) {
                const double rect1_x = rect1[0];
                const double rect1_y = rect1[1];
                const double rect1_w = rect1[2];
                const double rect1_h = rect1[3];

                const double rect2_x = rect2[0];
                const double rect2_y = rect2[1];
                const double rect2_w = rect2[2];
                const double rect2_h = rect2[3];

                const double left = rect1_x > rect2_x ? rect1_x : rect2_x;
                const double top = rect1_y > rect2_y ? rect1_y : rect2_y;
                double lhs = rect1_x + rect1_w;
                double rhs = rect2_x + rect2_w;
                const double right = lhs < rhs ? lhs : rhs;
                lhs = rect1_y + rect1_h;
                rhs = rect2_y + rect2_h;
                const double bottom = lhs < rhs ? lhs : rhs;

                Eigen::Vector4d rect_intersection;
                rect_intersection[0] = right < left ? 0 : left;
                rect_intersection[1] = bottom < top ? 0 : top;
                rect_intersection[2] = right < left ? 0 : right - left;
                rect_intersection[3] = bottom < top ? 0 : bottom - top;

                return rect_intersection;
            }

            double IntersectionOverUnion2d(const Eigen::Vector4d &rect1, const Eigen::Vector4d &rect2) {
                Eigen::Vector4d rect_intersection = Intersection2d(rect1, rect2);
                const double intersection_area = rect_intersection[2] * rect_intersection[3]; // Surface of the intersection of the rects
                const double union_of_rects = rect1[2] * rect1[3] + rect2[2] * rect2[3] - intersection_area; // Union of the area of the rects
                return intersection_area / union_of_rects; // Intersection over union
            }

            Eigen::Vector4d BoundingBox2d(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                          const std::vector<int> &indices, double percentage) {

                pcl::PointIndices filtered_indices = SUN::utils::filter::FilterPointCloudBasedOnRadius(scene_cloud, indices, percentage);

                int bbx_min = static_cast<int>(1e10);
                int bbx_max = static_cast<int>(-1e10);
                int bby_min = static_cast<int>(1e10);
                int bby_max = static_cast<int>(-1e10);

                for (auto ind:filtered_indices.indices) {
                    int x = -1, y = -1;
                    UnravelIndex(ind, scene_cloud->width, &x, &y);
                    if (x < bbx_min)
                        bbx_min = x;
                    if (x > bbx_max)
                        bbx_max = x;
                    if (y < bby_min)
                        bby_min = y;
                    if (y > bby_max)
                        bby_max = y;
                }

                // [min_x min_y w h]
                Eigen::Vector4d bb2d_out;
                bb2d_out[0] = static_cast<double>(bbx_min);
                bb2d_out[1] = static_cast<double>(bby_min);
                bb2d_out[2] = static_cast<double>(bbx_max - bbx_min);
                bb2d_out[3] = static_cast<double>(bby_max - bby_min);

                return bb2d_out;
            }

            Eigen::Vector4d BoundingBox2d(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr object_cloud, const SUN::utils::Camera &cam, double percentage) {
                auto filtered_cloud = SUN::utils::filter::FilterPointCloudBasedOnRadius(object_cloud,  percentage);

                int bbx_min = static_cast<int>(1e10);
                int bbx_max = static_cast<int>(-1e10);
                int bby_min = static_cast<int>(1e10);
                int bby_max = static_cast<int>(-1e10);

                for (const auto &pt:filtered_cloud->points) {
                    int x, y;

                    auto proj_pt = cam.CameraToImage(pt.getVector4fMap().cast<double>());
                    x = proj_pt[0];
                    y = proj_pt[1];

                    if (x < bbx_min)
                        bbx_min = x;
                    if (x > bbx_max)
                        bbx_max = x;
                    if (y < bby_min)
                        bby_min = y;
                    if (y > bby_max)
                        bby_max = y;
                }

                // [min_x min_y w h]
                Eigen::Vector4d bb2d_out;
                bb2d_out[0] = static_cast<double>(bbx_min);
                bb2d_out[1] = static_cast<double>(bby_min);
                bb2d_out[2] = static_cast<double>(bbx_max - bbx_min);
                bb2d_out[3] = static_cast<double>(bby_max - bby_min);

                return bb2d_out;
            }

            Eigen::VectorXd BoundingBox3d(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud_in, double percentage) {
                // Remove some points (radius-based cleaning)
                auto cloud_to_process = SUN::utils::filter::FilterPointCloudBasedOnRadius(cloud_in, percentage);

                // Compute mean, covariance matrix 3d
                Eigen::Matrix3d cov_mat3d;
                Eigen::Vector4d mean3d;
                pcl::computeMeanAndCovarianceMatrix(*cloud_to_process, cov_mat3d, mean3d);

                // Let's restrict ourselves to 2D ground-plane projection. More robust.
                Eigen::Matrix2d cov_mat2d;
                cov_mat2d(0, 0) = cov_mat3d(0, 0);
                cov_mat2d(1, 1) = cov_mat3d(2, 2);
                cov_mat2d(0, 1) = cov_mat3d(0, 2);
                cov_mat2d(1, 0) = cov_mat3d(2, 0);

                // Compute Eigen vectors, values.
                // Here, we get 2 eigenvectors, corresponding to dominant axes of 2D proj. of object (to the ground plane).
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver(cov_mat2d, Eigen::ComputeEigenvectors);
                Eigen::Matrix2d eigen_vectors = eigen_solver.eigenvectors();

                // Def. local coord. sys.
                Eigen::Matrix3d p2w(Eigen::Matrix3d::Identity());
                p2w.block<2, 2>(0, 0) = eigen_vectors.transpose();
                Eigen::Vector2d c_2d(mean3d[0], mean3d[2]); // Mean on gp. proj.
                p2w.block<2, 1>(0, 2) = -1.f * (p2w.block<2, 2>(0, 0) * c_2d); //centroid.head<2>());

                // Find bbox extent (width, depth)
                float bbx_min = static_cast<float>(1e4);
                float bbx_max = static_cast<float>(-1e4);
                float bbz_min = static_cast<float>(1e4);
                float bbz_max = static_cast<float>(-1e4);

                for (int i = 0; i <cloud_to_process->size(); i++) {

                    const auto &p_ref = cloud_to_process->at(i);
                    Eigen::Vector3d p_eig(p_ref.x, p_ref.z, 1.0);

                    if (std::isnan(p_ref.x))
                        continue;

                    p_eig = p2w * p_eig;

                    const float tmp_x = p_eig[0];
                    const float tmp_z = p_eig[1];

                    if (tmp_x < bbx_min)
                        bbx_min = tmp_x;
                    if (tmp_x > bbx_max)
                        bbx_max = tmp_x;
                    if (tmp_z < bbz_min)
                        bbz_min = tmp_z;
                    if (tmp_z > bbz_max)
                        bbz_max = tmp_z;
                }

                // Find out orientation
                Eigen::Matrix3d R_mat;
                R_mat.setIdentity();
                R_mat(0, 0) = eigen_vectors(0, 0);
                R_mat(0, 2) = eigen_vectors(0, 1);
                R_mat(2, 0) = eigen_vectors(1, 0);
                R_mat(2, 2) = eigen_vectors(1, 1);
                const Eigen::Vector2d mean_diag = 0.5f * (Eigen::Vector2d(bbx_min + bbx_max, bbz_min + bbz_max));
                const Eigen::Vector2d tfinal = eigen_vectors * mean_diag + Eigen::Vector2d(mean3d[0], mean3d[2]);

                // Final transform
                Eigen::Quaterniond qfinal(R_mat);
                double bb1 = bbx_max - bbx_min;
                double bb3 = bbz_max - bbz_min;

                // Get height
                Eigen::Vector4f cloud_min, cloud_max;
                pcl::getMinMax3D(*cloud_to_process, cloud_min, cloud_max);
                double min_y = cloud_min[1];
                double max_y = cloud_max[1];
                double cloud_height = std::abs(max_y - min_y); // Difference between Y-coords.

                // Resulting data structure
                Eigen::VectorXd bb3d_out;
                bb3d_out.setZero(10, 1); // center_x, center_y, center_z, width, height, depth, quaternion

                // X-Z center from 2d-gp-PCA, Y compute from 3D points
                bb3d_out(0) = tfinal[0];
                bb3d_out(1) = min_y + (max_y - min_y) / 2.0;
                bb3d_out(2) = tfinal[1];

                // Width, height, depth
                bb3d_out(3) = std::abs(bb1);
                bb3d_out(4) = std::abs(cloud_height);
                bb3d_out(5) = std::abs(bb3);

                // Quaternion, representing orientation
                //qfinal.setIdentity();
                bb3d_out(6) = qfinal.w();
                bb3d_out(7) = qfinal.x();
                bb3d_out(8) = qfinal.y();
                bb3d_out(9) = qfinal.z();

                return bb3d_out;
            }

            Eigen::VectorXd BoundingBox3d(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                          const std::vector<int> &indices, double percentage) {
                // Remove some points (radius-based cleaning)
                auto filtered_indices = SUN::utils::filter::FilterPointCloudBasedOnRadius(scene_cloud, indices, percentage);

                // Compute mean, covariance matrix 3d
                Eigen::Matrix3d cov_mat3d;
                Eigen::Vector4d mean3d;
                pcl::computeMeanAndCovarianceMatrix(*scene_cloud, indices, cov_mat3d, mean3d);

                // Let's restrict ourselves to 2D ground-plane projection. More robust.
                Eigen::Matrix2d cov_mat2d;
                cov_mat2d(0, 0) = cov_mat3d(0, 0);
                cov_mat2d(1, 1) = cov_mat3d(2, 2);
                cov_mat2d(0, 1) = cov_mat2d(1, 0) = cov_mat3d(0, 2);

                // Compute Eigen vectors, values..
                // Here, we get 2 eigenvectors, corresponding to dominant axes of 2D proj. of object (to the ground plane).
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver(cov_mat2d, Eigen::ComputeEigenvectors);
                Eigen::Matrix2d eigen_vectors = eigen_solver.eigenvectors();

                // Def. local coord. sys.
                Eigen::Matrix3d p2w(Eigen::Matrix3d::Identity());
                p2w.block<2, 2>(0, 0) = eigen_vectors.transpose();
                Eigen::Vector2d c_2d(mean3d[0], mean3d[2]); // Mean on gp. proj.
                p2w.block<2, 1>(0, 2) = -1.f * (p2w.block<2, 2>(0, 0) * c_2d); //centroid.head<2>());

                // Find bbox extent (width, depth)
                float bbx_min = 1e10;
                float bbx_max = -1e10;
                float bbz_min = 1e10;
                float bbz_max = -1e10;

                for (int i = 0; i < filtered_indices.indices.size(); i++) {
                    int ind = filtered_indices.indices.at(i);
                    const auto &p_ref = scene_cloud->at(ind);

                    if (std::isnan(p_ref.x))
                        continue;

                    Eigen::Vector3d p_eig(p_ref.x, p_ref.z, 1.0);
                    p_eig = p2w * p_eig;

                    const double tmp_x = p_eig[0];
                    const double tmp_z = p_eig[1];

                    if (tmp_x < bbx_min)
                        bbx_min = tmp_x;
                    if (tmp_x > bbx_max)
                        bbx_max = tmp_x;
                    if (tmp_z < bbz_min)
                        bbz_min = tmp_z;
                    if (tmp_z > bbz_max)
                        bbz_max = tmp_z;
                }

                // Find out orientation
                Eigen::Matrix3d R_mat;
                R_mat.setIdentity();
                R_mat(0, 0) = eigen_vectors(0, 0);
                R_mat(0, 2) = eigen_vectors(0, 1);
                R_mat(2, 0) = eigen_vectors(1, 0);
                R_mat(2, 2) = eigen_vectors(1, 1);
                const Eigen::Vector2d mean_diag = 0.5f * (Eigen::Vector2d(bbx_min + bbx_max, bbz_min + bbz_max));
                const Eigen::Vector2d tfinal = eigen_vectors * mean_diag + Eigen::Vector2d(mean3d[0], mean3d[2]);

                // Final transform
                const Eigen::Quaterniond qfinal(R_mat);
                double bb1 = bbx_max - bbx_min;
                double bb3 = bbz_max - bbz_min;

                // Get height
                Eigen::Vector4f cloud_min, cloud_max;
                pcl::getMinMax3D(*scene_cloud, filtered_indices, cloud_min, cloud_max);
                double min_y = cloud_min[1];
                double max_y = cloud_max[1];
                double cloud_height = std::abs(max_y - min_y); // Difference between Y-coords.

                // Resulting data structure
                Eigen::VectorXd bb3d_out;
                bb3d_out.setZero(10, 1); // center_x, center_y, center_z, width, height, depth, quaternion

                // X-Z center from 2d-gp-PCA, Y compute from 3D points
                bb3d_out(0) = tfinal[0];
                bb3d_out(1) = min_y + (max_y - min_y) / 2.0;
                bb3d_out(2) = tfinal[1];

                // Width, height, depth
                bb3d_out(3) = std::abs(bb1);
                bb3d_out(4) = std::abs(cloud_height);
                bb3d_out(5) = std::abs(bb3);

                // Quaternion, representing orientation
                bb3d_out(6) = qfinal.w();
                bb3d_out(7) = qfinal.x();
                bb3d_out(8) = qfinal.y();
                bb3d_out(9) = qfinal.z();

                return bb3d_out;
            }

            Eigen::Vector4d ReparametrizeBoundingBoxCenterMidToLeftTopPoint(const Eigen::Vector4d &bounding_box_2d) {
                auto cx = bounding_box_2d[0];
                auto cy = bounding_box_2d[1];
                auto w = bounding_box_2d[2];
                auto h = bounding_box_2d[3];
                return Eigen::Vector4d(cx - (w / 2.0), cy - (h / 2.0), w, h);
            }

            Eigen::Vector4d ReparametrizeBoundingBoxCenterTopToLeftTopPoint(const Eigen::Vector4d &bounding_box_2d) {
                auto cx = bounding_box_2d[0];
                auto cy = bounding_box_2d[1];
                auto w = bounding_box_2d[2];
                auto h = bounding_box_2d[3];
                return Eigen::Vector4d(cx - (w / 2.0), cy, w, h);
            }
            /**
               * @brief converts 3D bounding box with centriod, height, width, length to 8 corner bounding box. 
               *        Works only for box parallel to ground plane. Also, with assume no rotation except around y-axis.
               * @param point: the one centriod of the bounding box.
               * @param h: hieght of the bounding box.
               * @param w: width of the boudning box.
               * @param l: length of the bounding box.
               * @param rotationY: rotation around the y axis.
               * @return box3d: matrix with shape=[3,8], each column represents a corner
               * @author Omran kaddah.omran@gmail.com.
               */
            Eigen::MatrixXd Convert3DBoxTo8Corner(const Eigen::Vector3d &point, const double &h, const double &w, const double &l,const double rotationY){
                double c = std::cos(rotationY);
                double s = std::sin(rotationY);
                Eigen::Matrix3d R;
                R.setZero();
                R << c, 0, s,
                     0, 1, 0,
                    -s, 0, c;
                Eigen::MatrixXd box3d(3,8);
                box3d.row(0) << l/2,  l/2, - l/2, -l/2, l/2,  l/2,  -l/2,  -l/2;
                box3d.row(1) << 0  ,  0   ,  0  ,  0,   -h,   -h,   -h,      -h;
                box3d.row(2) << w/2, -w/2, -w/2,  w/2  ,w/2  ,-w/2,  -w/2,   w/2;
                box3d = R * box3d;
                box3d.row(0) = box3d.row(0) + Eigen::RowVectorXd::Constant(1,8,point[0]);
                box3d.row(1) = box3d.row(1) + Eigen::RowVectorXd::Constant(1,8,point[1]);
                box3d.row(2) = box3d.row(2) + Eigen::RowVectorXd::Constant(1,8,point[2]);
                return box3d;

            }

            /**
               * @brief converts 3D bounding box with centriod, height, width, length to 8 corner bounding box. 
               * @ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping
               *        Works only for box parallel to ground plane. Also, with assume no rotation except around y-axis.
               * @param point: the one centriod of the bounding box.
               * @param rotationY: rotation around the y axis.
               * @return box3d: matrix with shape=[3,8], each column represents a corner
               * @author Omran kaddah.omran@gmail.com.
               */
            double ConvexHullIntersection (std::vector<cv::Point2f> subjectPolygon, std::vector<cv::Point2f> clipPolygon) {
                auto inside = [](cv::Point2f p, cv::Point2f p1, cv::Point2f p2){
                    return (p2.y - p1.y) * p.x + (p1.x - p2.x) * p.y + (p2.x * p1.y - p1.x * p2.y) < 0;
                    };
                auto intersection = [](cv::Point2f cp1, cv::Point2f cp2, cv::Point2f s, cv::Point2f e){
                    cv::Point2f dc;
                    dc.x = cp1.x - cp2.x;
                    dc.y = cp1.y - cp2.y;
                    cv::Point2f dp;
                    dp.x = s.x - e.x;
                    dp.y =  s.y - e.y;
                    float n1 = cp1.x * cp2.y - cp1.y * cp2.x;
                    float n2 = s.x * e.y - s.y * e.x;
                    float n3 = 1.0 / (dc.x * dp.y - dc.y * dp.x);
                    cv::Point2f returned;
                    returned.x = (n1 * dp.x - n2 * dc.x) * n3;
                    returned.y = (n1 * dp.y - n2 * dc.y) * n3;
                    return returned;
                    };
                // ****************************
                // Sutherland-Hodgman clipping
                // ****************************
                int subjectPolygonSize = subjectPolygon.size();
                int clipPolygonSize = clipPolygon.size();
                std::vector<cv::Point2f> newPolygon = std::vector<cv::Point2f>(8);
                int newPolygonSize;
                cv::Point2f cp1, cp2, s, e;
                std::vector<cv::Point2f> inputPolygon = std::vector<cv::Point2f>(8);
            
                // copy subject polygon to new polygon and set its size
                for(int i = 0; i < subjectPolygonSize; i++)
                    newPolygon[i] = subjectPolygon[i];
                newPolygonSize = subjectPolygonSize;
            
                for(int j = 0; j < clipPolygonSize; j++)
                {
                    // copy new polygon to input polygon & set counter to 0
                    for(int k = 0; k < newPolygonSize; k++){ inputPolygon[k] = newPolygon[k]; }

                    int counter = 0;
            
                    // get clipping polygon edge
                    cp1 = clipPolygon[j];
                    cp2 = clipPolygon[(j + 1) % clipPolygonSize];
            
                    for(int i = 0; i < newPolygonSize; i++)
                    {
                        // get subject polygon edge
                        s = inputPolygon[i];
                        e = inputPolygon[(i + 1) % newPolygonSize];
            
                        // Case 1: Both vertices are inside:
                        // Only the second vertex is added to the output list
                        if(inside(s, cp1, cp2) && inside(e, cp1, cp2))
                            newPolygon[counter++] = e;
            
                        // Case 2: First vertex is outside while second one is inside:
                        // Both the point of intersection of the edge with the clip boundary
                        // and the second vertex are added to the output list
                        else if(!inside(s, cp1, cp2) && inside(e, cp1, cp2))
                        {
                            newPolygon[counter++] = intersection(cp1, cp2, s, e);
                            newPolygon[counter++] = e;
                        }
            
                        // Case 3: First vertex is inside while second one is outside:
                        // Only the point of intersection of the edge with the clip boundary
                        // is added to the output list
                        else if(inside(s, cp1, cp2) && !inside(e, cp1, cp2))
                            newPolygon[counter++] = intersection(cp1, cp2, s, e);
            
                        // Case 4: Both vertices are outside
                        else if(!inside(s, cp1, cp2) && !inside(e, cp1, cp2))
                        {
                            // No vertices are added to the output list
                        }
                    }
                    // set new polygon size
                    newPolygonSize = counter;
                }
                // ***********************************
                // End of Sutherland-Hodgman clipping
                // ***********************************
                if (newPolygonSize!= 0)
                {   
                    for( int remaining = 8 - newPolygonSize; remaining>0; --remaining )
                        newPolygon.pop_back();
                    std::vector<cv::Point2f> hull;
                    cv::convexHull(cv::Mat(newPolygon), hull, true);
                    double area = 0;
                    for (int i = 0; i < hull.size(); i++){
                        int next_i = (i+1)%(hull.size());
                        double dX   = hull[next_i].x - hull[i].x;
                        double avgY = (hull[next_i].y + hull[i].y)/2;
                        area += dX*avgY;  // This is the integration step.
                    }
                    // std::vector<cv::Point2f> contour;
                    // cv::approxPolyDP(cv::Mat(hull), contour, 0.001, true);
                    // double area2 = std::fabs(cv::contourArea(cv::Mat(contour)));
                    return std::fabs(area);
                }
                
                return 0;
            }

            /**
               * @brief Computes Intersection Over Union IoU in 3D. 
               *        Works only for box parallel to ground plane. Also, with assume no rotation except around y-axis.
               * @param corners2: matrix with shape=[3,8], each column represents a corner.
               * @param corners2: matrix with shape=[3,8], each column represents a corner.
               * @return IoU: double
               * @author Omran kaddah.omran@gmail.com.
               */
            double IntersectionOverUnion3d(const Eigen::MatrixXd &corners1, const Eigen::MatrixXd &corners2) {
                std::vector<cv::Point2f> rect1;
                std::vector<cv::Point2f> rect2;
                cv::Point2f pt;
                for( int i =3; i>=0; i--){
                    
                    pt.x = corners1(0,i);
                    pt.y = corners1(2,i);
                    rect1.push_back(pt);
                    pt.x = corners2(0,i);
                    pt.y = corners2(2,i);
                    rect2.push_back(pt);
                }
                double interArea = ConvexHullIntersection(rect1, rect2);
                double ymax = std::min(corners1(1,0), corners2(1,0));
                double ymin = std::max(corners1(1,4), corners2(1,4));
                double inter_vol = interArea * std::max(0.0, ymax-ymin);
                auto boxVol = [](const Eigen::MatrixXd & corners){
                    double volum = std::sqrt((corners.col(0) - corners.col(1)).array().pow(2).sum());
                    double volum1 = std::sqrt((corners.col(1) - corners.col(2)).array().pow(2).sum());
                    double volum2 = std::sqrt((corners.col(0) - corners.col(4)).array().pow(2).sum());
                    return volum * volum1 * volum2;
                };
       
                return inter_vol / (boxVol(corners1) + boxVol(corners2) - inter_vol);
            }
        }
    }
}